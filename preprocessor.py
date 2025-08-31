# preprocessor.py

import os
import cv2
import numpy as np
import pandas as pd

# --- Accelerometer Processing ---

def process_accelerometer_data(file_path, window_size=128, step=64):
    """
    Loads accelerometer data, calculates SMV, creates windows, and extracts features.
    
    Args:
        file_path (str): Path to the accelerometer CSV file.
        window_size (int): The number of time steps in each window.
        step (int): The number of time steps to slide the window forward.

    Returns:
        np.ndarray: A NumPy array of feature vectors for each window.
    """
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return np.array([])

    # Load the data
    df = pd.read_csv(file_path)
    
    # Keep only the relevant columns if others exist
    accel_cols = [col for col in df.columns if 'acc' in col.lower()]
    if len(accel_cols) < 3:
        # Fallback to column positions if names are not standard
        df = df.iloc[:, :3]
        df.columns = ['acc_x', 'acc_y', 'acc_z']
    else:
        df = df[accel_cols]
        df.columns = ['acc_x', 'acc_y', 'acc_z']

    # 1. Feature Engineering: Signal Magnitude Vector (SMV)
    df['smv'] = np.sqrt(df['acc_x']**2 + df['acc_y']**2 + df['acc_z']**2)

    # 2. Windowing
    windows = []
    for i in range(0, len(df) - window_size, step):
        window = df[i: i + window_size]
        
        # 3. Feature Extraction for the window
        features = {
            'x_mean': window['acc_x'].mean(),
            'y_mean': window['acc_y'].mean(),
            'z_mean': window['acc_z'].mean(),
            'smv_mean': window['smv'].mean(),
            'x_std': window['acc_x'].std(),
            'y_std': window['acc_y'].std(),
            'z_std': window['acc_z'].std(),
            'smv_std': window['smv'].std(),
        }
        windows.append(list(features.values()))
    
    return np.array(windows)


# --- Image Processing ---

# preprocessor.py - (REPLACE the image function with this FINAL version)

def process_image_sequence(dir_path, num_frames=32, target_size=(128, 128)):
    """
    Loads, resizes, and normalizes a sequence of images from a directory.
    This version is robust to hidden files (like .DS_Store) and an extra subdirectory.
    """
    if not os.path.isdir(dir_path):
        print(f"Directory not found: {dir_path}")
        return np.array([])

    # --- START of Updated Logic ---
    # Filter out hidden files (like .DS_Store on macOS)
    dir_contents = [item for item in os.listdir(dir_path) if not item.startswith('.')]

    # If there's only one visible item and it's a directory, go inside it
    if len(dir_contents) == 1 and os.path.isdir(os.path.join(dir_path, dir_contents[0])):
        dir_path = os.path.join(dir_path, dir_contents[0])
    # --- END of Updated Logic ---

    frame_files = sorted([os.path.join(dir_path, f) for f in os.listdir(dir_path) if f.endswith(('.png', '.jpg', '.jpeg'))])

    if len(frame_files) == 0:
        print(f"ERROR: Still no image files found in the final path: {dir_path}")
        return np.array([])

    # 1. Frame Selection
    total_frames = len(frame_files)
    indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    selected_files = [frame_files[i] for i in indices]

    frames = []
    for file_path in selected_files:
        # 2. Load and Resize
        frame = cv2.imread(file_path)
        frame = cv2.resize(frame, target_size)

        # 3. Normalize
        frame = frame.astype('float32') / 255.0
        frames.append(frame)

    return np.array(frames)