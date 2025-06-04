#!/usr/bin/env python3
"""
Database initialization script for enhanced vehicle classification system
"""

import os
import logging
from app import app, db
from models import PredictionHistory, BatchProcessing

def init_database():
    """Initialize database with enhanced schema"""
    with app.app_context():
        try:
            # Drop existing tables and recreate with new schema
            db.drop_all()
            db.create_all()
            
            logging.info("Database initialized successfully with enhanced schema")
            print("✓ Database initialized with enhanced vehicle classification schema")
            print("✓ New features: model tracking, processing time, confidence categories")
            print("✓ Support for 15 vehicle types")
            
        except Exception as e:
            logging.error(f"Error initializing database: {str(e)}")
            print(f"✗ Database initialization failed: {str(e)}")
            raise

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    init_database()