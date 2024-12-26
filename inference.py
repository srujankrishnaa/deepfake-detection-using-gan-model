from flask import Flask, request, render_template
from tensorflow.keras.models import load_model # type: ignore
from tensorflow.keras.preprocessing import image # type: ignore
import numpy as np
import os

class InferenceModel:
    """
    A class to load a trained model and handle file uploads for predictions.
    """

    def init(self, model_path):
        """
        Initialize the InferenceModel class.

        Args:
            model_path (str): Path to the saved Keras model.
        """
        self.model = load_model(model_path)
        self.app = Flask(_name_)
        self.app.config['UPLOAD_FOLDER'] = 'static/uploads'
        self.model_path = model_path

        @self.app.route('/', methods=['GET', 'POST'])
        #@self.app.route('/', methods=['GET', 'POST'])
        def upload_file():
            """
            Handle file upload and prediction requests.

            Returns:
            --------
            str
                The rendered HTML template with the result or error message.
            """
            if request.method == 'POST':
                if 'file' not in request.files:
                    return render_template('index.html', error='No file part.')
                file = request.files['file']
                if file.filename == '':
                    return render_template('index.html', error='No selected file.')
                if file and self.allowed_file(file.filename):
                    # Save the uploaded file to the uploads directory
                    filename = file.filename
                    filepath = os.path.join(self.app.config['UPLOAD_FOLDER'], filename)
                    file.save(filepath)
                    
                    # Predict the image
                    prediction, prediction_percentage = self.predict_image(filepath)
                    fake_percentage = round(100 - prediction_percentage, 2)
                    real_percentage = round(prediction_percentage, 2)
                    result = 'Real' if prediction >= 0.5 else 'Fake'
                    
                    # Pass image and results to the template
                    return render_template(
                        'index.html',
                        result=result,
                        real_percentage=real_percentage,
                        fake_percentage=fake_percentage,
                        uploaded_image=filename
                    )
                else:
                    return render_template('index.html', error='Allowed file types are png, jpg, jpeg.')
            return render_template('index.html')


    def allowed_file(self, filename):
        """
        Check if a file has an allowed extension.

        Parameters:
        -----------
        filename : str
            The name of the file to check.

        Returns:
        --------
        bool
            True if the file has an allowed extension, False otherwise.
        """
        # ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
        self.app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}
        return '.' in filename and filename.rsplit('.', 1)[1].lower() in self.app.config['ALLOWED_EXTENSIONS']

        #return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

    def predict_image(self, file_path):
        """
        Predict whether an image is Real or Fake using the loaded model.

        Parameters:
        -----------
        file_path : str
            The path to the image file.

        Returns:
        --------
        tuple
            A tuple containing the prediction and the prediction percentage.
        """
        img = image.load_img(file_path, target_size=(128, 128))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        result = self.model.predict(img_array)
        prediction = result[0][0]
        prediction_percentage = prediction * 100
        return prediction, prediction_percentage

    def run(self):
        """
        Run the Flask application with the loaded model.
        """
        self.app.run(debug=True)


if __name__ == 'main':
    # inference
    model_path = 'deepfake_detector_model_best.keras'
    inference_model = InferenceModel(model_path)
    inference_model.run()
