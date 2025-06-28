from ann_function import AEModel
from path_manager import PathManager
import logging
import os


class AEStart:
    def __init__(self, path_manager):
        """
        Initialize the TrainingManager with a PathManager instance.
        :param path_manager: Instance of PathManager
        """
        self.path_manager = path_manager

        # Paths managed by PathManager
        self.models_path = self.path_manager.get_path('model')
        self.params_path = self.path_manager.get_path('params')
        self.checkpoint_path = self.path_manager.get_path('checkpoint')
        self.log_path = self.path_manager.get_path('log')

        # Initialize AEModel
        self.ae_model = AEModel(self.path_manager)

        # Training state variables
        self.checkpoint = None
        self.epoch = 0
        self.image_count = 0

    def initialize(self):
        """
        Initialize the autoencoder model, training parameters, and checkpoint.
        """
        # Load the latest model if it exists
        model = self.ae_model.load_latest_model()

        if model is None:
            # Build a new model if no latest model exists
            input_shape = (128, 128, 1)  # Adjust this as per your dataset
            model = self.ae_model.build_model(input_shape)
            logging.info("Built a new model.")

        # Load the checkpoint if it exists
        self.checkpoint = self.ae_model.load_checkpoint()
        if self.checkpoint:
            self.epoch = self.checkpoint.get('epoch', 0)
            self.image_count = self.checkpoint.get('image_count', 0)
            logging.info(f"Resuming training from epoch {self.epoch}, image count {self.image_count}.")
        else:
            logging.info("No checkpoint found. Starting training from scratch.")

    def start_training(self):
        """
        Start the training process.
        """
        logging.info("Training started.")
        # Insert the actual training logic here, e.g., loading data, training loops, etc.
        # Update checkpoint periodically and save progress
        logging.info("Training completed.")

    def save_training_progress(self):
        """
        Save the training progress and model at the current checkpoint.
        """
        checkpoint = {
            'epoch': self.epoch,
            'image_count': self.image_count,
        }
        self.ae_model.save_checkpoint(checkpoint)
        self.ae_model.save_latest_model()
        logging.info(f"Training progress saved at epoch {self.epoch}, image count {self.image_count}.")


if __name__ == "__main__":
    from path_manager import PathManager

    # Initialize PathManager
    path_manager = PathManager(base_path='./dataset')

    # Initialize TrainingManager
    training_manager = TrainingManager(path_manager)

    # Initialize the training setup
    training_manager.initialize()

    # Start the training process
    training_manager.start_training()

