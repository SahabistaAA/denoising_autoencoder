import os
import logging

class PathManager:
    def __init__(self, base_path='./dataset'):
        self.base_path = base_path
        self.dataset_path = os.path.join(self.base_path, 'dataset')
        self.models_path = os.path.join(self.base_path, 'model')
        self.params_path = os.path.join(self.base_path, 'params')
        self.checkpoint_path = os.path.join(self.base_path, 'checkpoint')
        self.log_path = os.path.join(self.base_path, 'log')

        # Initialize directories
        self._create_directories()

        # Set up logging
        self._setup_logging()

    def _create_directories(self):
        """Create required directories if they don't exist."""
        for path in [
            self.dataset_path,
            self.params_path,
            self.checkpoint_path,
            self.log_path
        ]:
            os.makedirs(path, exist_ok=True)

    def _setup_logging(self):
        """Configure the logging settings."""
        logging.basicConfig(
            filename=os.path.join(self.log_path, 'ann_data_training.log'),
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )

    def get_path(self, key):
        """Retrieve a specific path by key."""
        paths = {
            'dataset': self.dataset_path,
            'model': self.models_path,
            'params': self.params_path,
            'checkpoint': self.checkpoint_path,
            'log': self.log_path
        }
        return paths.get(key, None)
    

# Example Usage
if __name__ == '__main__':
    path_manager = PathManager(base_path='./my_project')

    # Access paths
    dataset_path = path_manager.get_path('dataset')
    log_path = path_manager.get_path('log')

    print(f"Dataset Path: {dataset_path}")
    print(f"Log Path: {log_path}")

    # Log an example message
    logging.info("Paths and logging have been initialized.")