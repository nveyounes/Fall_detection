import numpy as np
import tensorflow as tf
import os
from preprocessor import process_accelerometer_data, process_image_sequence

class FallDetectionDataGenerator(tf.keras.utils.Sequence):
    """
    Custom Data Generator to feed fall detection data to a Keras model.
    It loads and preprocesses image sequences and accelerometer data in batches.
    """
    def __init__(self, data_dir, event_folders, labels, batch_size=16,
                 image_dim=(128, 128), n_channels=3, n_frames=32,
                 accel_window_size=128, accel_step=64, shuffle=True):
        """Initialization"""
        self.data_dir = data_dir
        self.event_folders = event_folders
        self.labels = labels
        self.batch_size = batch_size
        self.image_dim = image_dim
        self.n_channels = n_channels
        self.n_frames = n_frames
        self.accel_window_size = accel_window_size
        self.accel_step = accel_step
        self.shuffle = shuffle
        self.indexes = np.arange(len(self.event_folders))
        self.on_epoch_end()

    def __len__(self):
        """Denotes the number of batches per epoch"""
        return int(np.floor(len(self.event_folders) / self.batch_size))

    def __getitem__(self, index):
        """Generate one batch of data"""
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of event folder names
        batch_event_folders = [self.event_folders[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(batch_event_folders)

        return X, y

    def on_epoch_end(self):
        """Updates indexes after each epoch"""
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, batch_event_folders):
        """Generates data containing batch_size samples"""
        # Initialization
        # Note: We'll have two inputs: images and accelerometer features
        X_images = np.empty((self.batch_size, self.n_frames, *self.image_dim, self.n_channels))
        X_accel = [] # Using a list because the number of windows can vary
        y = np.empty((self.batch_size), dtype=int)

        # Generate data for each event in the batch
        for i, folder_name in enumerate(batch_event_folders):
            label = self.labels[folder_name]
            event_type = 'falls' if label == 1 else 'adls'
            event_path = os.path.join(self.data_dir, event_type, folder_name)

            # --- Process Image Data ---
            # We'll use cam0's RGB data for this model
            image_path = os.path.join(event_path, 'cam0', 'rgb')
            X_images[i,] = process_image_sequence(image_path, self.n_frames, self.image_dim)

            # --- Process Accelerometer Data ---
            accel_path = os.path.join(event_path, 'acc', f"{folder_name}-acc.csv")
            accel_features = process_accelerometer_data(accel_path, self.accel_window_size, self.accel_step)
            # For simplicity, we'll average the features across all windows for one event
            if accel_features.shape[0] > 0:
                 X_accel.append(np.mean(accel_features, axis=0))
            else:
                 # If no features, append a zero vector of the correct shape (8 features)
                 X_accel.append(np.zeros(8))


            y[i] = label

        # We return two inputs for the model, and the corresponding labels
        return [np.array(X_images), np.array(X_accel)], np.array(y)