import h5py
import numpy as np
import tensorflow as tf


def load_h5_data(train_x_path, train_y_path, val_x_path, val_y_path,
                 num_train=20000, num_val=5000):
    """
    Load train and validation datasets from HDF5 files.

    Uses a configurable subset size to fit within memory constraints.
    The full PCam training set is 262,144 images (96x96x3) which requires
    ~27 GB of RAM when loaded as float32.

    Args:
        train_x_path: path to training images .h5 file
        train_y_path: path to training labels .h5 file
        val_x_path: path to validation images .h5 file
        val_y_path: path to validation labels .h5 file
        num_train: number of training samples to load (None = all)
        num_val: number of validation samples to load (None = all)

    Returns:
        X_train, y_train, X_val, y_val as numpy arrays
    """
    # Load training data
    with h5py.File(train_x_path, "r") as f:
        if num_train is not None:
            X_train = np.array(f["x"][:num_train])
        else:
            X_train = np.array(f["x"])

    with h5py.File(train_y_path, "r") as f:
        if num_train is not None:
            y_train = np.array(f["y"][:num_train])
        else:
            y_train = np.array(f["y"])

    # Load validation data
    with h5py.File(val_x_path, "r") as f:
        if num_val is not None:
            X_val = np.array(f["x"][:num_val])
        else:
            X_val = np.array(f["x"])

    with h5py.File(val_y_path, "r") as f:
        if num_val is not None:
            y_val = np.array(f["y"][:num_val])
        else:
            y_val = np.array(f["y"])

    print(f"Train images: {X_train.shape}  Train labels: {y_train.shape}")
    print(f"Val images:   {X_val.shape}    Val labels:   {y_val.shape}")

    # Normalize pixel values to [0, 1]
    X_train = X_train.astype("float32") / 255.0
    X_val = X_val.astype("float32") / 255.0

    # Flatten label dimensions: (N, 1, 1, 1) -> (N,)
    y_train = y_train.reshape(-1)
    y_val = y_val.reshape(-1)

    return X_train, y_train, X_val, y_val


def load_test_data(test_x_path, test_y_path, num_test=5000):
    """
    Load test dataset from HDF5 files.

    Args:
        test_x_path: path to test images .h5 file
        test_y_path: path to test labels .h5 file
        num_test: number of test samples to load (None = all)

    Returns:
        X_test, y_test as numpy arrays
    """
    with h5py.File(test_x_path, "r") as f:
        if num_test is not None:
            X_test = np.array(f["x"][:num_test])
        else:
            X_test = np.array(f["x"])

    with h5py.File(test_y_path, "r") as f:
        if num_test is not None:
            y_test = np.array(f["y"][:num_test])
        else:
            y_test = np.array(f["y"])

    print(f"Test images: {X_test.shape}  Test labels: {y_test.shape}")

    X_test = X_test.astype("float32") / 255.0
    y_test = y_test.reshape(-1)

    return X_test, y_test


def build_augmentation_layer():
    """
    Build a data augmentation pipeline using Keras preprocessing layers.

    Augmentations applied:
        - Random horizontal flip
        - Random vertical flip
        - Random rotation (up to 20 degrees)
        - Random brightness adjustment
        - Random contrast adjustment

    These augmentations reflect common variations seen in whole-slide
    imaging: tissue orientation is arbitrary (flips, rotations), and
    staining intensity varies across labs (brightness, contrast).
    """
    return tf.keras.Sequential([
        layers.RandomFlip("horizontal_and_vertical"),
        layers.RandomRotation(0.06),  # ~20 degrees
    ], name="augmentation")


# Need to import layers for the augmentation function
from tensorflow.keras import layers


def create_augmented_dataset(X, y, batch_size=64, augment=True):
    """
    Create a tf.data.Dataset with optional augmentation.

    Args:
        X: numpy array of images
        y: numpy array of labels
        batch_size: batch size for training
        augment: whether to apply data augmentation

    Returns:
        tf.data.Dataset
    """
    dataset = tf.data.Dataset.from_tensor_slices((X, y))

    if augment:
        aug_layer = build_augmentation_layer()

        def augment_fn(image, label):
            image = aug_layer(image, training=True)
            # Random brightness
            image = tf.image.random_brightness(image, max_delta=0.1)
            # Random contrast
            image = tf.image.random_contrast(image, lower=0.9, upper=1.1)
            # Clip values to valid range
            image = tf.clip_by_value(image, 0.0, 1.0)
            return image, label

        dataset = dataset.map(augment_fn, num_parallel_calls=tf.data.AUTOTUNE)

    dataset = dataset.shuffle(buffer_size=min(len(X), 10000))
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset
