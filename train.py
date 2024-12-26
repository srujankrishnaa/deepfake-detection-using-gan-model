import os
import shutil
import urllib3
import requests
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

class DatasetHandler:
    """
    A class to handle dataset downloading, unzipping, loading, and processing.
    """

    def __init__(self, dataset_url, dataset_download_dir, dataset_file, dataset_dir, train_dir, test_dir, val_dir):
        self.dataset_url = dataset_url
        self.dataset_download_dir = dataset_download_dir
        self.dataset_file = dataset_file
        self.dataset_dir = dataset_dir
        self.train_dir = os.path.join(dataset_dir, train_dir)
        self.test_dir = os.path.join(dataset_dir, test_dir)
        self.val_dir = os.path.join(dataset_dir, val_dir)

    def download_dataset(self):
        os.makedirs(self.dataset_download_dir, exist_ok=True)
        file_path = os.path.join(self.dataset_download_dir, self.dataset_file)
        if os.path.exists(file_path):
            print(f"Dataset file {self.dataset_file} already exists at {file_path}")
            return True
        try:
            response = requests.get(self.dataset_url, stream=True, timeout=60)
            total_size = int(response.headers.get("content-length", 0))
            with open(file_path, "wb") as file, tqdm(
                desc=self.dataset_file, total=total_size, unit="iB", unit_scale=True, unit_divisor=1024
            ) as bar:
                for data in response.iter_content(chunk_size=1024):
                    size = file.write(data)
                    bar.update(size)
            print(f"Dataset downloaded and saved to {file_path}")
            return True
        except requests.RequestException as e:
            print(f"Error during dataset download: {e}")
            return False

    def unzip_dataset(self):
       file_path = os.path.join(self.dataset_download_dir, self.dataset_file)
       if os.path.exists(self.dataset_dir):
            print(f"Dataset already extracted at {self.dataset_dir}")
            return True
       if not os.path.exists(file_path):
            print(f"Dataset file {file_path} not found after download")
            return False
       try:
            shutil.unpack_archive(file_path, self.dataset_download_dir)
            print(f"Dataset extracted to {self.dataset_dir}")
            return True
       except(shutil.ReadError, FileNotFoundError) as e:
            print(f"Error during dataset extraction: {e}")
            return False

    def get_image_dataset_from_directory(self, dir_name, augment=False):
        dir_path = os.path.join(self.dataset_dir, dir_name)
        data_augmentation = tf.keras.Sequential([  
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.2),
            layers.RandomZoom(0.2),
            layers.RandomBrightness(0.2)
        ]) if augment else None
        dataset = tf.keras.utils.image_dataset_from_directory(
            dir_path,
            labels='inferred',
            color_mode='rgb',
            seed=42,
            batch_size=64,
            image_size=(128, 128),
        )
        return dataset.map(lambda x, y: (data_augmentation(x) if augment else x, y))

    def load_split_data(self):
        train_data = self.get_image_dataset_from_directory(self.train_dir, augment=True)
        test_data = self.get_image_dataset_from_directory(self.test_dir)
        val_data = self.get_image_dataset_from_directory(self.val_dir)
        return train_data, test_data, val_data

class GAN:
    """
    A simple GAN implementation for generating synthetic images.
    """

    def __init__(self, latent_dim=100):
        self.latent_dim = latent_dim  # Define latent_dim
        self.generator = self._build_generator()
        self.discriminator = self._build_discriminator()
        self.gan = self._build_gan()

    def _build_generator(self):
        model = tf.keras.Sequential([
            layers.Dense(256, activation='relu', input_dim=self.latent_dim),  # Use latent_dim here
            layers.BatchNormalization(),
            layers.LeakyReLU(),
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.LeakyReLU(),
            layers.Dense(128 * 128 * 3, activation='tanh'),
            layers.Reshape((128, 128, 3))
        ])
        return model

    def _build_discriminator(self):
        model = tf.keras.Sequential([
            layers.Flatten(input_shape=(128, 128, 3)),
            layers.Dense(512, activation='relu'),
            layers.LeakyReLU(),
            layers.Dense(256, activation='relu'),
            layers.LeakyReLU(),
            layers.Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
        return model

    def _build_gan(self):
        self.discriminator.trainable = False
        model = tf.keras.Sequential([
            self.generator,
            self.discriminator
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy')
        return model

    def train(self, train_data, epochs, batch_size):
        for epoch in range(epochs):
            print(f"Epoch {epoch + 1}/{epochs}")
            for real_images, _ in train_data.take(100):
                noise = tf.random.normal([batch_size, self.latent_dim])
                fake_images = self.generator.predict(noise)
                fake_labels = np.zeros((batch_size, 1))
                real_labels = np.ones((real_images.shape[0], 1))
                d_loss_real = self.discriminator.train_on_batch(real_images, real_labels)
                d_loss_fake = self.discriminator.train_on_batch(fake_images, fake_labels)
                misleading_labels = np.ones((batch_size, 1))
                g_loss = self.gan.train_on_batch(noise, misleading_labels)
            print(
                f"D Loss (real): {d_loss_real}, D Loss (fake): {d_loss_fake}, G Loss: {g_loss}"
            )

    def generate_images(self, num_images):
        noise = np.random.normal(0, 1, (num_images, self.latent_dim))  # Use latent_dim here
        generated_images = self.generator.predict(noise)
        return (generated_images + 1) / 2.0  # Rescale to [0, 1]

class DeepfakeDetectorModel:
    """
    A class to create and train a deepfake detection model.
    """

    def __init__(self):
        self.model = self._build_model()

    def _build_model(self):
        base_model = tf.keras.applications.ResNet50(
            input_shape=(128, 128, 3),
            include_top=False,
            weights='imagenet'
        )
        base_model.trainable = True  # Enable fine-tuning

        model = models.Sequential([
            tf.keras.layers.InputLayer(input_shape=(128, 128, 3)),
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dropout(0.5),
            layers.Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
            layers.Dense(1, activation='sigmoid')
        ])
        return model

    def compile_model(self, learning_rate=0.0001):
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss='binary_crossentropy', 
            metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall(), tf.keras.metrics.AUC()],
        )

    def train_model(self, train_data, val_data, epochs, class_weights):
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, min_lr=1e-7, verbose=1),
            ModelCheckpoint('deepfake_detector_model_best.keras', monitor='val_loss', save_best_only=True, verbose=1)
        ]
        history =  self.model.fit(
            train_data,
            validation_data=val_data,
            epochs=epochs,
            callbacks=callbacks,
            class_weight=class_weights
        )
        return history

    def evaluate_model(self, test_data):
        evaluation_results = self.model.evaluate(test_data, return_dict=True)
        y_true = np.concatenate([y for x, y in test_data], axis=0)
        y_pred_probs = self.model.predict(test_data).flatten()
        y_pred = (y_pred_probs > 0.5).astype(int)
        cm = confusion_matrix(y_true, y_pred)
        report = classification_report(y_true, y_pred, target_names=['Real', 'Deepfake'], output_dict=True)
        evaluation_results.update({
            'confusion_matrix': cm,
            'f1_score': report['weighted avg']['f1-score'],
            'precision': report['weighted avg']['precision'],
            'recall': report['weighted avg']['recall']
        })
        return evaluation_results

    def save_model(self, path):
        self.model.save(path)

def evaluate_saved_model(test_data, model_path='deepfake_detector_model_best.keras'):
    model = tf.keras.models.load_model(model_path)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall(), tf.keras.metrics.AUC()]
    )
    evaluation_results = model.evaluate(test_data, return_dict=True)
    y_true = np.concatenate([y for x, y in test_data], axis=0)
    y_pred_probs = model.predict(test_data).flatten()
    y_pred = (y_pred_probs > 0.5).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=['Real', 'Deepfake'], output_dict=True)
    evaluation_results.update({
        'confusion_matrix': cm,
        'f1_score': report['weighted avg']['f1-score'],
        'precision': report['weighted avg']['precision'],
        'recall': report['weighted avg']['recall']
    })
    return evaluation_results

if __name__ == '__main__':
    dataset_url = 'https://www.kaggle.com/api/v1/datasets/download/manjilkarki/deepfake-and-real-images?datasetVersionNumber=1'
    dataset_download_dir = './data'
    dataset_file = 'dataset.zip'
    dataset_dir = './data/Dataset'
    train_dir = 'Train'
    test_dir = 'Test'
    val_dir = 'Validation'

    dataset_handler = DatasetHandler(
        dataset_url, dataset_download_dir, dataset_file, dataset_dir, train_dir, test_dir, val_dir
    )

    if not dataset_handler.download_dataset():
        print('Failed to download dataset')
    if not dataset_handler.unzip_dataset():
        print('Failed to unzip dataset')

    train_data, test_data, val_data = dataset_handler.load_split_data()

    # Train GAN and generate synthetic data
    gan = GAN()
    gan.train(train_data,epochs = 100,batch_size = 64)
    synthetic_images = gan.generate_images(1000)

    # Integrate synthetic images into training data
    synthetic_labels = tf.constant([1] * 1000)  # Assume synthetic images are labeled as "Deepfake"
    synthetic_dataset = tf.data.Dataset.from_tensor_slices((synthetic_images, synthetic_labels)).batch(64)
    train_data = train_data.concatenate(synthetic_dataset)

    class_weights = {0: 1.0, 1: 1.0}  # Adjust based on dataset imbalance

    model = DeepfakeDetectorModel()
    model.compile_model(learning_rate=0.0001)

    history = model.train_model(train_data, val_data, epochs=50, class_weights=class_weights)
    model.save_model('deepfake_detector_model_best.keras')

    # Evaluate the saved model
    _, test_data, _ = dataset_handler.load_split_data()
    evaluation_metrics = evaluate_saved_model(test_data, model_path='deepfake_detector_model_best.keras')

    print("\nFinal Evaluation Metrics:")
    for key, value in evaluation_metrics.items():
        print(f"{key}: {value}")







    
