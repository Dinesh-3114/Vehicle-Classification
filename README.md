🚗 Vehicle Classification System
This project is a deep learning-based vehicle classification system that uses a fine-tuned ResNet50 model to classify images of vehicles into predefined categories such as cars, trucks, buses, and bikes. A Flask web application provides a simple interface for users to upload images and receive real-time classification results.

🧠 Model Details
Architecture: ResNet50 (pretrained on ImageNet)

Training: Fine-tuned on a custom-labeled vehicle dataset

Libraries: PyTorch, Torchvision

Performance: Achieved 92% test accuracy

🌐 Web Application
Backend: Flask handles image uploads and runs model inference

Frontend: HTML/CSS interface for uploading images and displaying predictions

Output: Vehicle type prediction with confidence score, returned instantly

🔧 Features
Upload vehicle images directly through the web interface

Real-time prediction with high accuracy

End-to-end pipeline from image input to prediction output

Training notebook included for model customization and retraining
