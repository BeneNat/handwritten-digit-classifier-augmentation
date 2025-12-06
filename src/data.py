from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

def load_data(normalize=True, flatten=False):
    """
    Loads and preprocesses the MNIST dataset.
    """
    # Load raw data from Keras
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Reshape to (N, 28, 28, 1) for CNN input
    x_train = x_train.reshape(-1, 28, 28, 1)
    x_test = x_test.reshape(-1, 28, 28, 1)

    # Normalize pixel values to [0, 1] range
    if normalize:
        x_train = x_train.astype("float32") / 255.0
        x_test = x_test.astype("float32") / 255.0

    # Flatten for non-convolutional models
    if flatten:
        x_train = x_train.reshape(len(x_train), -1)
        x_test = x_test.reshape(len(x_test), -1)

    # Convert labels to one-hot encoding vectors
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)

    return x_train, y_train, x_test, y_test
