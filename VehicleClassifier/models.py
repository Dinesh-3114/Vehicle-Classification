from app import db
from datetime import datetime
import json

class PredictionHistory(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(255), nullable=False)
    predicted_class = db.Column(db.String(100), nullable=False)
    confidence = db.Column(db.Float, nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    batch_id = db.Column(db.String(100), nullable=True)
    
    # Enhanced fields for improved accuracy tracking
    model_used = db.Column(db.String(50), default='tensorflow')
    all_predictions = db.Column(db.Text, nullable=True)  # JSON string of all class probabilities
    image_size = db.Column(db.String(20), nullable=True)  # Original image dimensions
    processing_time = db.Column(db.Float, nullable=True)  # Time taken for prediction
    confidence_category = db.Column(db.String(20), nullable=True)  # high, medium, low
    
    def __repr__(self):
        return f'<PredictionHistory {self.filename}: {self.predicted_class} ({self.confidence:.2f}) via {self.model_used}>'
    
    def get_all_predictions_dict(self):
        """Parse all_predictions JSON string to dictionary"""
        if self.all_predictions:
            try:
                return json.loads(self.all_predictions)
            except json.JSONDecodeError:
                return {}
        return {}
    
    def set_all_predictions_dict(self, predictions_dict):
        """Set all_predictions from dictionary"""
        self.all_predictions = json.dumps(predictions_dict)
    
    def get_confidence_category(self):
        """Determine confidence category based on confidence score"""
        if self.confidence >= 0.8:
            return 'high'
        elif self.confidence >= 0.6:
            return 'medium'
        else:
            return 'low'

class BatchProcessing(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    batch_id = db.Column(db.String(100), unique=True, nullable=False)
    total_images = db.Column(db.Integer, nullable=False)
    processed_images = db.Column(db.Integer, default=0)
    status = db.Column(db.String(50), default='processing')  # processing, completed, failed
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f'<BatchProcessing {self.batch_id}: {self.processed_images}/{self.total_images}>'
