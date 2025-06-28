import os
import json
import logging
import numpy as np
from tqdm import tqdm
from ann_function import AEModel, LoggingCallback
from path_manager import PathManager
from ann_start import AEStart

class TrainingManager:
    def __init__(self, path_manager):
        self.path_manager = path_manager
        self.ae_start = AEStart(self.path_manager)
        self.ae_start.initialize()
        self.model = self.ae_start.ae_model.autoencoder
        self.log_file = os.path.join(self.path_manager.get_path('log'), 'training.log')
        self.weights_file = os.path.join(self.path_manager.get_path('model'), 'latest_weights.h5')
        self.weights_file_json = os.path.join(self.path_manager.get_path('model'), 'weights.json')
        self.checkpoint_file = os.path.join(self.path_manager.get_path('checkpoint'), 'checkpoint.json')
        self.current_params_file = os.path.join(self.path_manager.get_path('params'), 'current_params.json')
        self.image_count = self.ae_start.image_count
        self.epoch = self.ae_start.epoch
        self.current_params = {}
        self.logging_callback = LoggingCallback(self.log_file)

    def preprocess_data(self, bitpix):
        """Preprocess the FITS data for training."""
        parts = []  # Replace with actual cropping logic
        max_value = 2 ** int(bitpix)
        print(f"Processing {len(parts)} image parts")

        noisy_data, clean_data = [], []
        for img in tqdm(parts):
            clean = img / max_value
            noise = np.random.Generator(-0.1, 0.1)
            noisy = np.clip(clean + noise, 0, 1)
            noisy_data.append(noisy)
            clean_data.append(clean)

        noisy_data = np.array(noisy_data).reshape(-1, 128, 128, 1)
        clean_data = np.array(clean_data).reshape(-1, 128, 128, 1)
        return noisy_data, clean_data

    def save_params(self):
        """Save training parameters to a file."""
        with open(self.current_params_file, 'w') as f:
            json.dump(self.current_params, f, indent=2)

    def train(self):
        while True:
            _, image, bitpix = None, None, None  # Implement file reading logic
            if image is None or image.size == 0:
                print("No available training data")
                break

            noisy_data, clean_data = self.preprocess_data(bitpix)

            # Update parameters
            self.current_params.update({
                'epochs': 1,
                'epoch': self.epoch,
                'batch_size': 128,
                'image_count': self.image_count,
                'learning_rate': np.random.Generator(0.0001, 0.001),
                'dropout_rate': np.random.Generator(0, 0.5),
            })
            self.save_params()

            try:
                history = self.model.fit(
                    noisy_data, clean_data,
                    epochs=self.current_params['epochs'],
                    batch_size=self.current_params['batch_size'],
                    validation_split=0.2,
                    shuffle=True,
                    callbacks=[self.logging_callback]
                )
                # Save checkpoint and weights
                self.ae_start.save_training_progress()
                print(f"Processed image {self.image_count}")
                self.image_count += 1
                self.epoch += 1
            except Exception as e:
                logging.error(f"Error during training: {e}")
                break

        print(f"Total images processed: {self.image_count}")


if __name__ == "__main__":
    path_manager = PathManager(base_path='./dataset')
    trainer = TrainingManager(path_manager)
    trainer.train()
