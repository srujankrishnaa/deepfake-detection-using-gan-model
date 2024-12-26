from flask import Flask, request, render_template, flash
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import logging
from werkzeug.utils import secure_filename
from typing import Tuple, Optional, Dict, Any

class InferenceModel:
    """Handles deepfake detection using a pre-trained model."""
    
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
    
    def _init_(self, model_path: str):
        """Initialize the inference model and Flask application.
        
        Args:
            model_path (str): Path to the trained model file
        """
        self.setup_logging()
        self.model = self.load_model(model_path)
        
        # Initialize Flask app
        self.app = Flask(__name__)
        self.app.config.update(
            UPLOAD_FOLDER='static/uploads',
            MAX_CONTENT_LENGTH=16 * 1024 * 1024,  # 16MB max file size
            SECRET_KEY='your-secret-key-here'  # Change this in production
        )
        
        # Ensure upload directory exists
        os.makedirs(self.app.config['UPLOAD_FOLDER'], exist_ok=True)
        
        # Register routes
        self.register_routes()
    
    def setup_logging(self) -> None:
        """Configure logging for the application."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    def load_model(self, model_path: str) -> Any:
        """Load the pre-trained model with error handling.
        
        Args:
            model_path (str): Path to the model file
            
        Returns:
            The loaded model or None if loading fails
        """
        try:
            model = load_model(model_path)
            self.logger.info("Model loaded successfully")
            return model
        except Exception as e:
            self.logger.error(f"Failed to load model: {str(e)}")
            return None

    def register_routes(self) -> None:
        """Register Flask routes."""
        self.app.add_url_rule('/', 
                             view_func=self.upload_file, 
                             methods=['GET', 'POST'])

    def allowed_file(self, filename: str) -> bool:
        """Check if the file has a valid extension.
        
        Args:
            filename (str): Name of the uploaded file
            
        Returns:
            bool: True if file extension is allowed, False otherwise
        """
        return ('.' in filename and 
                filename.rsplit('.', 1)[1].lower() in self.ALLOWED_EXTENSIONS)

    def predict_image(self, file_path: str) -> Tuple[float, float]:
        """Perform prediction on the uploaded image.
        
        Args:
            file_path (str): Path to the image file
            
        Returns:
            Tuple[float, float]: Prediction result and confidence percentage
            
        Raises:
            ValueError: If image processing fails
        """
        try:
            # Load and preprocess the image
            img = image.load_img(file_path, target_size=(128, 128))
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = img_array / 255.0  # Normalize the image
            
            # Make prediction
            result = self.model.predict(img_array)
            prediction = float(result[0][0])
            prediction_percentage = prediction * 100
            
            self.logger.info(f"Prediction made successfully: {prediction_percentage}%")
            return prediction, prediction_percentage
            
        except Exception as e:
            self.logger.error(f"Error in image prediction: {str(e)}")
            raise ValueError(f"Failed to process image: {str(e)}")

    def upload_file(self):
        """Handle file upload and image prediction.
        
        Returns:
            str: Rendered template with results or error messages
        """
        context: Dict[str, Any] = {
            'uploaded_image': None,
            'error': None
        }

        if request.method == 'POST':
            try:
                # Validate file existence
                if 'file' not in request.files:
                    raise ValueError('No file part in the request.')
                
                file = request.files['file']
                if file.filename == '':
                    raise ValueError('No file selected.')
                
                if not self.allowed_file(file.filename):
                    raise ValueError('Allowed file types are png, jpg, jpeg.')
                
                if self.model is None:
                    raise ValueError('Model not loaded. Please check server configuration.')
                
                # Save and process file
                filename = secure_filename(file.filename)
                filepath = os.path.join(self.app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)
                
                # Make prediction
                prediction, prediction_percentage = self.predict_image(filepath)
                
                # Prepare results
                context.update({
                    'result': 'Real' if prediction >= 0.5 else 'Fake',
                    'real_percentage': round(prediction_percentage, 2),
                    'fake_percentage': round(100 - prediction_percentage, 2),
                    'uploaded_image': filename
                })
                
            except Exception as e:
                self.logger.error(f"Error processing upload: {str(e)}")
                context['error'] = str(e)
        
        return render_template('index.html', **context)

    def run(self, host: str = '0.0.0.0', port: int = 5000, debug: bool = False) -> None:
        """Run the Flask application.
        
        Args:
            host (str): Host address to run the server on
            port (int): Port number to run the server on
            debug (bool): Whether to run in debug mode
        """
        self.logger.info(f"Starting server on {host}:{port}")
        self.app.run(host=host, port=port, debug=debug)

if __name__== '_main_':
    # Path to the deepfake detector model
    MODEL_PATH = 'deepfake_detector_model_best.keras'
    
    # Initialize and run the app
    inference_model = InferenceModel(MODEL_PATH)
    inference_model.run(debug=True)