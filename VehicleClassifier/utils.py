import os
from app import db
from models import PredictionHistory

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'}

def allowed_file(filename):
    """Check if file has allowed extension"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def save_prediction_to_db(filename, predicted_class, confidence, batch_id=None):
    """Save prediction result to database"""
    try:
        prediction = PredictionHistory(
            filename=filename,
            predicted_class=predicted_class,
            confidence=confidence,
            batch_id=batch_id
        )
        db.session.add(prediction)
        db.session.commit()
        return True
    except Exception as e:
        db.session.rollback()
        print(f"Error saving prediction to database: {str(e)}")
        return False

def save_prediction_to_db_enhanced(filename, result, batch_id=None):
    """Save enhanced prediction result to database"""
    try:
        prediction = PredictionHistory(
            filename=filename,
            predicted_class=result['predicted_class'],
            confidence=result['confidence'],
            batch_id=batch_id,
            model_used=result.get('model_used', 'enhanced_ml'),
            image_size=result.get('image_size', 'unknown'),
            processing_time=result.get('processing_time', 0.0),
            confidence_category=result.get('confidence_category', 'medium')
        )
        
        # Save all predictions as JSON
        if 'all_predictions' in result:
            prediction.set_all_predictions_dict(result['all_predictions'])
        
        db.session.add(prediction)
        db.session.commit()
        return True
    except Exception as e:
        db.session.rollback()
        print(f"Error saving enhanced prediction to database: {str(e)}")
        return False

def format_confidence(confidence):
    """Format confidence score as percentage"""
    return f"{confidence * 100:.1f}%"

def get_confidence_color(confidence):
    """Get Bootstrap color class based on confidence level"""
    if confidence >= 0.8:
        return 'success'
    elif confidence >= 0.6:
        return 'warning'
    else:
        return 'danger'

def format_file_size(size_bytes):
    """Format file size in human readable format"""
    if size_bytes == 0:
        return "0B"
    
    size_names = ["B", "KB", "MB", "GB"]
    i = 0
    while size_bytes >= 1024 and i < len(size_names) - 1:
        size_bytes /= 1024.0
        i += 1
    
    return f"{size_bytes:.1f}{size_names[i]}"
