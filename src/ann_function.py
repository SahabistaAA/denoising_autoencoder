import os
import json
import logging
import numpy as np
import tensorflow as tf
import random as r
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from datetime import datetime
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.metrics import BinaryAccuracy
from tensorflow.image import ssim
from path_manager import PathManager


class SSIMMetric():
    def __init__(self, threshold=0.9):
        self.threshold = threshold
    
    def __call__(self, y_true, y_pred):
        return tf.reduce_mean(ssim(y_true, y_pred, max_val=1.0), axis=[1, 2, 3])

class AEModel:
    def __init__(self, path_manager):
        """
        Initializes AEModel with a PathManager instance.
        """
        self.path_manager = path_manager
        self.models_path = self.path_manager.get_path('models')
        self.params_path = self.path_manager.get_path('params')

    def build_model(self, input_shape=(128, 128, 1)):
        """
        Builds and compiles the autoencoder model.
        """
        # Encoder
        input_layer = Input(shape=input_shape)
        x = Conv2D(32, (3, 3), activation='sigmoid', padding='same')(input_layer)
        x = MaxPooling2D((2, 2), padding='same')(x)
        x = Conv2D(64, (3, 3), activation='sigmoid', padding='same')(x)
        x = MaxPooling2D((2, 2), padding='same')(x)
        x = Conv2D(128, (3, 3), activation='sigmoid', padding='same')(x)
        encoded = MaxPooling2D((2, 2), padding='same')(x)

        # Decoder
        x = Conv2D(128, (3, 3), activation='sigmoid', padding='same')(encoded)
        x = UpSampling2D((2, 2))(x)
        x = Conv2D(64, (3, 3), activation='sigmoid', padding='same')(x)
        x = UpSampling2D((2, 2))(x)
        x = Conv2D(32, (3, 3), activation='sigmoid', padding='same')(x)
        x = UpSampling2D((2, 2))(x)
        decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

        # Compile
        self.autoencoder = Model(input_layer, decoded)
        self.autoencoder.compile(
            optimizer=Adam(),
            loss=BinaryCrossentropy()
        )
        self.autoencoder.summary()
        return self.autoencoder

    def save_model(self, params):
        """
        Saves the model, weights, and parameters with timestamped filenames.
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_filename = f"model_{timestamp}.keras"
        weights_filename = f"weights_{timestamp}.h5"
        params_filename = f"params_{timestamp}.json"

        try:
            # Save model
            self.autoencoder.save(os.path.join(self.models_path, model_filename))

            # Save weights
            self.autoencoder.save_weights(os.path.join(self.models_path, weights_filename))

            # Save parameters
            with open(os.path.join(self.params_path, params_filename), 'w') as f:
                json.dump(params, f, indent=2)

            logging.info(f"Model, weights, and parameters saved with timestamp: {timestamp}")
        except Exception as e:
            logging.error(f"Error saving model: {str(e)}")

    def save_latest_model(self):
        """
        Saves the current model as the latest model.
        """
        latest_model_filename = "latest_model.keras"
        try:
            self.autoencoder.save(os.path.join(self.models_path, latest_model_filename))
            logging.info("Latest model saved successfully.")
        except Exception as e:
            logging.error(f"Error saving latest model: {str(e)}")

    def load_latest_model(self):
        """
        Loads the most recent model saved as "latest_model.keras".
        """
        latest_model_filename = "latest_model.keras"
        model_path = os.path.join(self.models_path, latest_model_filename)
        if os.path.exists(model_path):
            try:
                self.autoencoder = tf.keras.models.load_model(model_path)
                logging.info("Latest model loaded successfully.")
                return self.autoencoder
            except Exception as e:
                logging.error(f"Error loading latest model: {str(e)}")
        else:
            logging.warning("No latest model found.")
            return None

    def save_checkpoint(self, checkpoint):
        """
        Saves a checkpoint to track training progress.
        """
        checkpoint_filename = os.path.join(self.models_path, "checkpoint.json")
        try:
            with open(checkpoint_filename, 'w') as f:
                json.dump(checkpoint, f, indent=2)
            logging.info("Checkpoint saved successfully.")
        except Exception as e:
            logging.error(f"Error saving checkpoint: {str(e)}")

    def load_checkpoint(self):
        """
        Loads the training checkpoint.
        """
        checkpoint_filename = os.path.join(self.models_path, "checkpoint.json")
        if os.path.exists(checkpoint_filename):
            try:
                with open(checkpoint_filename, 'r') as f:
                    checkpoint = json.load(f)
                logging.info("Checkpoint loaded successfully.")
                return checkpoint
            except Exception as e:
                logging.error(f"Error loading checkpoint: {str(e)}")
        else:
            logging.warning("No checkpoint found.")
            return None

class LoggingCallback(Callback):
    def __init__(self, log_file):
        super().__init__()
        self.log_file = log_file

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        log_entry = {
            'epoch': epoch,
            'loss': logs.get('loss'),
            'val_loss': logs.get('val_loss'),
            'timestamp': datetime.now().isoformat()
        }
        try:
            with open(self.log_file, 'a') as f:
                json.dump(log_entry, f)
                f.write('\n')
            logging.info(f"Epoch {epoch} logs saved.")
        except Exception as e:
            logging.error(f"Error saving epoch log: {str(e)}")

            