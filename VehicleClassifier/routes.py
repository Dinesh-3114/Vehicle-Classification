import os
import uuid
import time
from flask import render_template, request, jsonify, flash, redirect, url_for, session
from werkzeug.utils import secure_filename
from app import app, db
from models import PredictionHistory, BatchProcessing
from simple_enhanced_classifier import SimpleEnhancedClassifier
from utils import allowed_file, save_prediction_to_db
import logging

# Initialize the enhanced classifier
classifier = SimpleEnhancedClassifier()

@app.route('/')
def index():
    """Main page with upload interface"""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_files():
    """Handle file upload and prediction"""
    try:
        if 'files' not in request.files:
            flash('No files selected', 'error')
            return redirect(url_for('index'))
        
        files = request.files.getlist('files')
        
        if not files or all(file.filename == '' for file in files):
            flash('No files selected', 'error')
            return redirect(url_for('index'))
        
        # Create batch ID for tracking
        batch_id = str(uuid.uuid4())
        
        # Filter valid files
        valid_files = [f for f in files if f and allowed_file(f.filename)]
        
        if not valid_files:
            flash('No valid image files found', 'error')
            return redirect(url_for('index'))
        
        # Create batch record
        batch_record = BatchProcessing(
            batch_id=batch_id,
            total_images=len(valid_files),
            status='processing'
        )
        db.session.add(batch_record)
        db.session.commit()
        
        # Save uploaded files and prepare for prediction
        uploaded_files = []
        for file in valid_files:
            if file:
                filename = secure_filename(file.filename)
                unique_filename = f"{uuid.uuid4()}_{filename}"
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
                file.save(file_path)
                uploaded_files.append({
                    'path': file_path,
                    'original_name': filename
                })
        
        # Perform enhanced predictions
        results = []
        for i, file_info in enumerate(uploaded_files):
            try:
                prediction_result = classifier.predict_single_image(file_info['path'])
                
                # Save enhanced prediction to database
                save_prediction_to_db(
                    filename=file_info['original_name'],
                    predicted_class=prediction_result['predicted_class'],
                    confidence=prediction_result['confidence'],
                    batch_id=batch_id
                )
                
                # Add to results with enhanced information
                results.append({
                    'filename': file_info['original_name'],
                    'predicted_class': prediction_result['predicted_class'],
                    'confidence': prediction_result['confidence'],
                    'all_predictions': prediction_result['all_predictions'],
                    'model_used': prediction_result.get('model_used', 'enhanced_ml'),
                    'processing_time': prediction_result.get('processing_time', 0.0),
                    'confidence_category': prediction_result.get('confidence_category', 'medium'),
                    'status': 'success'
                })
                
                # Update batch progress
                batch_record.processed_images = i + 1
                db.session.commit()
                
            except Exception as e:
                logging.error(f"Error processing {file_info['original_name']}: {str(e)}")
                results.append({
                    'filename': file_info['original_name'],
                    'status': 'error',
                    'error': str(e)
                })
            
            finally:
                # Clean up uploaded file
                try:
                    os.remove(file_info['path'])
                except:
                    pass
        
        # Update batch status
        batch_record.status = 'completed'
        db.session.commit()
        
        # Store results in session for display
        session['results'] = results
        session['batch_id'] = batch_id
        
        return redirect(url_for('show_results'))
        
    except Exception as e:
        logging.error(f"Error in upload_files: {str(e)}")
        flash(f'Error processing files: {str(e)}', 'error')
        return redirect(url_for('index'))

@app.route('/results')
def show_results():
    """Display prediction results"""
    results = session.get('results', [])
    batch_id = session.get('batch_id', None)
    
    if not results:
        flash('No results to display', 'info')
        return redirect(url_for('index'))
    
    # Calculate statistics
    successful_predictions = [r for r in results if r.get('status') == 'success']
    total_files = len(results)
    successful_files = len(successful_predictions)
    
    # Group by vehicle type
    vehicle_counts = {}
    for result in successful_predictions:
        vehicle_type = result['predicted_class']
        vehicle_counts[vehicle_type] = vehicle_counts.get(vehicle_type, 0) + 1
    
    statistics = {
        'total_files': total_files,
        'successful_files': successful_files,
        'failed_files': total_files - successful_files,
        'vehicle_counts': vehicle_counts,
        'average_confidence': sum(r['confidence'] for r in successful_predictions) / len(successful_predictions) if successful_predictions else 0
    }
    
    return render_template('results.html', 
                         results=results, 
                         statistics=statistics,
                         batch_id=batch_id)

@app.route('/history')
def view_history():
    """View prediction history"""
    page = request.args.get('page', 1, type=int)
    per_page = 20
    
    # Get recent predictions
    predictions = PredictionHistory.query.order_by(
        PredictionHistory.timestamp.desc()
    ).paginate(
        page=page,
        per_page=per_page,
        error_out=False
    )
    
    # Get batch statistics
    batch_stats = db.session.query(
        BatchProcessing.batch_id,
        BatchProcessing.total_images,
        BatchProcessing.timestamp,
        BatchProcessing.status
    ).order_by(BatchProcessing.timestamp.desc()).limit(10).all()
    
    return render_template('history.html', 
                         predictions=predictions,
                         batch_stats=batch_stats)

@app.route('/api/model-info')
def get_model_info():
    """Get information about the model"""
    try:
        model_info = classifier.get_model_info()
        return jsonify(model_info)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/batch-status/<batch_id>')
def get_batch_status(batch_id):
    """Get status of batch processing"""
    try:
        batch = BatchProcessing.query.filter_by(batch_id=batch_id).first()
        if not batch:
            return jsonify({'error': 'Batch not found'}), 404
        
        return jsonify({
            'batch_id': batch.batch_id,
            'total_images': batch.total_images,
            'processed_images': batch.processed_images,
            'status': batch.status,
            'progress': (batch.processed_images / batch.total_images) * 100 if batch.total_images > 0 else 0
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/clear-session')
def clear_session():
    """Clear session data"""
    session.clear()
    flash('Session cleared', 'info')
    return redirect(url_for('index'))

@app.errorhandler(413)
def too_large(e):
    flash('File is too large. Maximum file size is 16MB.', 'error')
    return redirect(url_for('index'))

@app.errorhandler(404)
def not_found(e):
    return render_template('404.html'), 404

@app.errorhandler(500)
def server_error(e):
    return render_template('500.html'), 500
