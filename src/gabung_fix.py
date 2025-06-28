import os
import json
import logging
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import random as r
from astropy.io import fits
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

def read_fits_count(directory):
    files = os.listdir(directory)

    for file in files:
        if (file.endswith(".fits") or file.endswith(".fit")):
            file_path = os.path.join(directory, file)
            print("FITS file detected:", file_path)

            # Open the FITS file
            '''hdu_list = fits.open(file_path)
            image_data = hdu_list[0].data
            header = hdu_list[0].header
            hdu_list.close()'''
            with fits.open(file_path) as hdu_list:
                image_data = hdu_list[0].data
                bitpix = hdu_list[0].header['BITPIX']

            # after saving the data, delete the original fits file
            #os.remove(file_path)
            #print(f"Original FITS file deleted: {file_path}")

            return file_path, image_data, bitpix

    # If no FITS file is found, return None
    return None, None, None

'''def read_fits_count(directory):
    files = os.listdir(directory)

    for file in files:
        if (file.endswith(".fits") or file.endswith(".fit")):
            file_path = os.path.join(directory, file)
            print("FITS file detected:", file_path)

            # Open the FITS file
            hdu_list = fits.open(file_path)
            image_data = hdu_list[0].data
            header = hdu_list[0].header
            hdu_list.close()

            # after saving the data, delete the original fits file
            os.remove(file_path)
            print(f"Original FITS file deleted: {file_path}")

            return image_data, header['BITPIX']

    # If no FITS file is found, return None
    return None, None'''
# def crop_and_extract(fits_file, crop_size=128, overlap_threshold=0.2):
#     # Load the FITS data
#     hdulist = fits.open(fits_file)
#     data = hdulist[0].data

#     # Calculate the overlap in pixels
#     overlap = int(crop_size * overlap_threshold)

#     # Calculate the number of crops in each dimension
#     num_crops_x = int(np.ceil((data.shape[0] - crop_size) / (crop_size - overlap)) + 1)
#     num_crops_y = int(np.ceil((data.shape[1] - crop_size) / (crop_size - overlap)) + 1)

#     # List to store the cropped data
#     cropped_data = []


#         # Crops iterations
#     for i in range(num_crops_x):
#         for j in range(num_crops_y):
#             # Calculate the starting coordinates of the crop
#             start_x = i * (crop_size - overlap)
#             start_y = j * (crop_size - overlap)

#             # Calculate the ending coordinates of the crop
#             end_x = min(start_x + crop_size, data.shape[0])
#             end_y = min(start_y + crop_size, data.shape[1])
#             # Handle edge cases where the crop goes beyond the image boundaries
#             if end_x > data.shape[0]:
#                 end_x = data.shape[0]  # Adjust end_x to the image edge
#                 start_x = end_x - crop_size # Ensure the crop is still 128 pixels wide
#             if end_y > data.shape[1]:
#                 end_y = data.shape[1]  # Adjust end_y to the image edge
#                 start_y = end_y - crop_size # Ensure the crop is still 128 pixels tall


#             # Extract the crop
#             crop = data[start_x:end_x, start_y:end_y]

#             # Append the crop to the list
#             if crop.shape == (512, 512):
#                 cropped_data.append(crop)

#     return cropped_data

def crop_and_extract(fits_file, crop_size=512, overlap_threshold=0.2):
    # Load the FITS data
    hdulist = fits.open(fits_file)
    data = hdulist[0].data

    # Calculate the overlap in pixels
    overlap = int(crop_size * overlap_threshold)

    # Calculate the number of crops in each dimension
    num_crops_x = int(np.ceil((data.shape[0] - crop_size) / (crop_size - overlap)) + 1)
    num_crops_y = int(np.ceil((data.shape[1] - crop_size) / (crop_size - overlap)) + 1)

    # List to store the cropped data
    cropped_data = []


        # Crops iterations
    for i in range(num_crops_x):
        for j in range(num_crops_y):
            # Calculate the starting coordinates of the crop
            start_x = i * (crop_size - overlap)
            start_y = j * (crop_size - overlap)

            # Calculate the ending coordinates of the crop
            end_x = start_x + crop_size
            end_y = start_y + crop_size
            # Handle edge cases where the crop goes beyond the image boundaries
            if end_x > data.shape[0]:
                end_x = data.shape[0]  # Adjust end_x to the image edge
                start_x = end_x - crop_size # Ensure the crop is still 512 pixels wide
            if end_y > data.shape[1]:
                end_y = data.shape[1]  # Adjust end_y to the image edge
                start_y = end_y - crop_size # Ensure the crop is still 128 pixels tall

            # Extract the crop
            crop = data[start_x:end_x, start_y:end_y]

            # Ensure the crop is exactly 128x128
            if crop.shape == (512, 512):
                cropped_data.append(crop)
            else:
                print(f"Skipping crop with shape {crop.shape}")

    return cropped_data

def gaussian_kernel(size, sigma=1):

    ax = np.linspace(-(size // 2), size // 2, size)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    return kernel / np.sum(kernel)

def add_cosmic_ray_noise(image, position, length, angle_degrees, intensity=1, width=1, falloff=1):
    angle_radians = np.radians(angle_degrees)
    delta_x = np.cos(angle_radians)
    delta_y = np.sin(angle_radians)

    image_with_ray = image.copy()
    start_y, start_x = position
    kernel_size = 2 * width + 1
    gaussian = gaussian_kernel(kernel_size, sigma=width / 2)

    for i in range(length):
        new_y = int(start_y + i * delta_y)
        new_x = int(start_x + i * delta_x)
        if 0 <= new_y < image.shape[0] and 0 <= new_x < image.shape[1]:
            # Apply intensity with falloff
            ray_intensity = intensity * (falloff ** i)

            for ky in range(-width, width + 1):
                for kx in range(-width, width + 1):
                    kernel_value = gaussian[ky + width, kx + width]
                    y = new_y + ky
                    x = new_x + kx
                    if 0 <= y < image.shape[0] and 0 <= x < image.shape[1]:
                        image_with_ray[y, x] = np.clip(image_with_ray[y, x] + ray_intensity * kernel_value, 0, 255)

    return image_with_ray
def generate_noise(snr_linear, shape=(128, 128)):
    noise_power = 1 / snr_linear
    noise = np.random.normal(0, (noise_power), shape)
    #print(noise_power)
    return noise

def randomnoise():
    number_CR = max(r.randint(-20,3),0)
    snr = r.randint(100,200)
    noise_sample = generate_noise(snr)
    for i in range(number_CR):
        cosmic_ray_position = (r.randint(10,120), r.randint(10,120))  
        cosmic_ray_length = r.randint(10,100)
        cosmic_ray_slope = r.randint(-90,90)
        noise_sample = add_cosmic_ray_noise(noise_sample, cosmic_ray_position, cosmic_ray_length, cosmic_ray_slope)
    return noise_sample


# Custom SSIM Metric
def ssim_metric(y_true, y_pred):
    return tf.reduce_mean(ssim(y_true, y_pred, max_val=1.0))

# Build model
def build_model(input_shape=(None, 128, 128, 1)):
    # Encoder
    input_img = Input(shape=input_shape)
    x = Conv2D(32, (3, 3), activation='sigmoid', padding='same')(input_img)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(32, (3, 3), activation='sigmoid', padding='same')(x)
    encoded = MaxPooling2D((2, 2), padding='same')(x)

    # Decoder
    x = Conv2D(32, (3, 3), activation='sigmoid', padding='same')(encoded)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(32, (3, 3), activation='sigmoid', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

    # Build and compile model
    autoencoder = Model(input_img, decoded)
    autoencoder.compile(optimizer=Adam(),
                        loss=BinaryCrossentropy(),
                        metrics=[ssim_metric]
                        )
    autoencoder.summary()

    return autoencoder

# Path for data saving
base_path = './dataset'
dataset_path = os.path.join(base_path, 'dataset')
models_path = os.path.join(base_path, 'model')
params_path = os.path.join(base_path, 'params')
checkpoint_path = os.path.join(base_path, 'checkpoint')
log_path = os.path.join(base_path, 'log')

for path in [dataset_path, params_path, checkpoint_path, log_path]:
    os.makedirs(path, exist_ok=True)

logging.basicConfig(
    filename=os.path.join(log_path, 'ann_data_training.log'),
    level = logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Function to save model and parameters
def save_model_params(model, params, models_path):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_filename = f"model_{timestamp}.keras"
    weights_filename = f"weights_{timestamp}.weights.h5"
    params_filename = f"params_{timestamp}.json"

    try:
        # Save model
        model.save(os.path.join(models_path, model_filename))

        # Save weights
        model.save_weights(os.path.join(params_path, weights_filename))

        # Save parameters
        with open(os.path.join(params_path, params_filename), 'w') as f:
            json.dump(params, f, indent=2)

        # Save latest model
        save_latest_model(model, models_path)

        logging.info(f"Model saved to {model_filename}, weights saved to {weights_filename}, and parameters saved to {params_filename}")
    except Exception as e:
        logging.error(f"Error saving model and parameters: {str(e)}")

# Function to save the latest model
def save_latest_model(model, models_path):
    latest_model_filename = "latest_model.keras"
    model.save(os.path.join(models_path, latest_model_filename))
    logging.info(f"Latest model saved to {latest_model_filename}")

def save_params(params, filename):
    with open(filename, 'w') as f:
        json.dump(params, f, indent=2)

# Function to load the latest model
def load_latest_model(models_path):
    latest_model_filename = "latest_model.keras"
    model_path = os.path.join(models_path, latest_model_filename)
    if os.path.exists(model_path):
        model = tf.keras.models.load_model(model_path)
        logging.info(f"Loaded latest model from {latest_model_filename}")
        return model
    else:
        logging.info("No latest model found.")
        return None

# Function to load model and parameters
def load_model_params(models_path):
    try:
        model_files = [f for f in os.listdir(models_path) if f.startswith('model_') and f.endswith('.keras')]
        if not model_files:
            logging.info("No model files found.")
            return None, None

        latest_model = max(model_files, key=lambda x: os.path.getctime(os.path.join(models_path, x)))
        model = tf.keras.models.load_model(os.path.join(models_path, latest_model))

        params_file = latest_model.replace('model_', 'params_').replace('.keras', '.json')
        with open(os.path.join(params_path, params_file), 'r') as f:
            params = json.load(f)

        logging.info(f"Loaded model: {latest_model} and corresponding parameters")
        return model, params
    except Exception as e:
        logging.error(f"Error loading model and parameters: {str(e)}")
        return None, None

def load_params(param_file):
    if os.path.exists(param_file):
        with open(param_file, 'r') as f:
            return json.load(f)
    return None

# Function for saving checkpoint to handle error
def save_checkpoint(checkpoint, checkpoint_path):
    try:
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        with open(checkpoint_path, 'w') as f:
            json.dump(checkpoint, f, indent=2)
        logging.info(f"Checkpoint saved to {checkpoint_path}")
    except Exception as e:
        logging.error(f"Error saving checkpoint: {str(e)}")
'''def save_checkpoint(model, epoch, loss, checkpoint_path):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_filename = f"checkpoint_{timestamp}.keras"
    checkpoint_info_filename = f"checkpoint_info_{timestamp}.json"

    try:
        # Save checkpoint
        model.save(os.path.join(checkpoint_path, checkpoint_filename))

        # Save checkpoint info
        checkpoint_info = {
            'epoch': epoch,
            'loss': loss,
            'model_file': checkpoint_filename
        }
        with open(os.path.join(checkpoint_path, checkpoint_info_filename), 'w') as f:
                json.dump(checkpoint_info, f, indent=2)

        logging.info(f"Checkpoint saved to {checkpoint_filename} and checkpoint info saved to {checkpoint_info_filename}")
    except Exception as e:
        logging.error(f"Error saving checkpoint: {str(e)}")'''

# Function for updating checkpoint
'''def update_checkpoint(checkpoint, file_path):
    existing_checkpoint, _ = load_model_params(file_path)
    if existing_checkpoint is None:
        existing_checkpoint = {}
    existing_checkpoint.update(checkpoint)
    save_model_params(existing_checkpoint, file_path, models_path)'''
'''def update_checkpoint(model, epoch, loss, checkpoint_path):
    # This function now just calls save_checkpoint
    save_checkpoint(model, epoch, loss, checkpoint_path)'''
'''def update_checkpoint(new_data, checkpoint_path):
    checkpoint = load_checkpoint(checkpoint_path)
    checkpoint.update(new_data)
    save_checkpoint(checkpoint, checkpoint_path)'''
def update_checkpoint(new_data, checkpoint_path):
    checkpoint = load_checkpoint(checkpoint_path)
    if isinstance(checkpoint, tuple):
        checkpoint = checkpoint[0] if checkpoint[0] is not None else {}
    checkpoint.update(new_data)
    save_checkpoint(checkpoint, checkpoint_path)

# Function for loading checkpoint
def load_checkpoint(checkpoint_path):
    try:
        if not os.path.exists(checkpoint_path):
            logging.error(f"Checkpoint path {checkpoint_path} does not exist.")
            return None, None

        checkpoint_files = [f for f in os.listdir(checkpoint_path) if f.startswith('checkpoint_') and f.endswith('.keras')]
        if not checkpoint_files:
            logging.info("No checkpoint files found.")
            return None, None

        latest_checkpoint = max(checkpoint_files, key=lambda x: os.path.getctime(os.path.join(checkpoint_path, x)))
        checkpoint_info_file = latest_checkpoint.replace('checkpoint_', 'checkpoint_info_').replace('.keras', '.json')
        if not os.path.exists(os.path.join(checkpoint_path, checkpoint_info_file)):
            logging.error(f"Checkpoint info file {checkpoint_info_file} not found.")
            return None, None

        model = tf.keras.models.load_model(os.path.join(checkpoint_path, latest_checkpoint))

        with open(os.path.join(checkpoint_path, checkpoint_info_file), 'r') as f:
            checkpoint_info = json.load(f)

        logging.info(f"Loaded checkpoint: {latest_checkpoint}")
        return model, checkpoint_info
    except Exception as e:
        logging.error(f"Error loading checkpoint: {str(e)}")
        return None, None

# Converting weights to json
def weights_to_json(model, filename):
    weights_list = []
    for layer in model.layers:
        layer_weights = layer.get_weights()
        if layer_weights:
            weights_list.append({
                'layer_name': layer.name,
                'weights': [w.tolist() for w in layer_weights]
            })

    with open(filename, 'w') as f:
        json.dump(weights_list, f)

# Converting json to weights
def json_to_weights(model, filename):
    with open(filename, 'r') as f:
        weights_list = json.load(f)

    for layer_weights in weights_list:
        layer_name = layer_weights['layer_name']
        if layer_name in [layer.name for layer in model.layers]:
            layer = model.get_layer(layer_name)
            layer.set_weights([np.array(w) for w in layer_weights['weights']])
        else:
            print(f"Skipping weights for non-existent layer: {layer_name}")

# Logging callback
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
        with open(self.log_file, 'a') as f:
            json.dump(log_entry, f)
            f.write('\n')

def read_log(log_file):
    with open(log_file, 'r') as f:
        for line in f:
            yield json.loads(line)

# Directory
directory = os.path.join(base_path,  'dataset')
current_params_file = os.path.join(params_path, 'current_params.json')
checkpoint_file = os.path.join(checkpoint_path, 'checkpoint.json')
all_params_file = os.path.join(params_path, 'all_params.json')
weights_file = os.path.join(params_path, 'model_weights.weights.h5')
weights_file_json = os.path.join(params_path, 'model_weights.json')

# Log
log_dir = os.path.join(log_path, 'logs')
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f'training_log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
logging_callback = LoggingCallback(log_file)

# Load the latest model and parameters
model, loaded_params = load_model_params(models_path)

if model is None:
    # If no model was loaded, build a new one
    input_shape = (128, 128, 1)  # adjust based on your image size
    model = build_model(input_shape)
    current_params = load_params(current_params_file)
    print("Built new model")
else:
    current_params = loaded_params
    print("Loaded latest model and parameters")

# Ensure current_params is a dictionary
if current_params is None:
    current_params = {}


# Initialize or update parameters
all_params = load_params(all_params_file)
image_count = current_params.get('image_count', 0)
epoch = current_params.get('epoch', 0)

# If weights file exists and no model was loaded, load weights
if os.path.exists(weights_file_json) and loaded_params is None:
    json_to_weights(model, weights_file_json)
    print("Loaded model weights from file")

print(f"Starting training from image {image_count} and epoch {epoch}")

while True:
    file_path, image, bitpix = read_fits_count(directory)
    if image is None or image.size == 0:
        print('No available training data')
        break

    parts = crop_and_extract(file_path)
    max_value = 2**int(bitpix)
    print(f'Processing {len(parts)} image parts')

    noisy_data = []
    clean_data = []

    for img in tqdm(parts):
        clean = img / max_value
        noise = randomnoise()
        noisy = np.clip(clean + noise, 0, 1)

        noisy_data.append(noisy)
        clean_data.append(clean)

    noisy_data = np.array(noisy_data).reshape(-1, 128, 128, 1)
    clean_data = np.array(clean_data).reshape(-1, 128, 128, 1)

    # Update current parameters
    current_params['epochs'] = 1
    current_params['epoch'] = epoch
    current_params['batch_size'] = 128
    current_params['image_count'] = image_count
    current_params['max_value'] = max_value
    current_params['learning_rate'] = r.uniform(0.0001, 0.001)
    current_params['l1_reg'] = r.uniform(0, 0.01)
    current_params['l2_reg'] = r.uniform(0, 0.01)
    current_params['dropout_rate'] = r.uniform(0, 0.5)

    # Save current parameters
    save_params(current_params, current_params_file)

    # Train the modell
    try:
        history = model.fit(
            noisy_data, clean_data,
            epochs=current_params['epochs'],
            batch_size=current_params['batch_size'],
            validation_split=0.2,
            shuffle=True,
            callbacks=[logging_callback]
        )

        # Save checkpoint
        checkpoint = {
            f'epoch_{epoch}': {
                'images_processed': image_count,
                'last_params': current_params,
                'final_loss': history.history['loss'][-1],
                'final_ssim': history.history['ssim_metric'][-1],
                'final_val_loss': history.history['val_loss'][-1],
                'final_val_ssim': history.history['val_ssim_metric'][-1]
                }
            }

        update_checkpoint(checkpoint, checkpoint_file)

        model.save_weights(weights_file)
        weights_to_json(model, weights_file_json)

        # Print training results
        for i in range(len(history.history['loss'])):
            print(f"Epoch {i+1}: accuracy = {history.history['ssim_metric'][i]}, "
                    f"loss = {history.history['loss'][i]}, "
                    f"val_accuracy = {history.history['val_ssim_metric'][i]}",
                    f"val_loss = {history.history['val_loss'][i]}")

    except Exception as e:
        print(f"An error occurred during training: {str(e)}")
        print("Skipping this image and moving to the next one.")
        #continue
        break

    image_count += 1
    epoch += 1

    print(f"Processed image {image_count}")
    print(f"Training log saved to: {log_file}")
    print(f"Checkpoint saved to: {checkpoint_file}")


print(f"Total images processed: {image_count}")