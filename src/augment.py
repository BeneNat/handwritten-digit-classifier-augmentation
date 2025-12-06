from tensorflow.keras.preprocessing.image import ImageDataGenerator

def get_augmenter():
    """
    Creates an ImageDataGenerator for real-time data augmentation.
    """
    augmenter = ImageDataGenerator(
        rotation_range = 10,
        width_shift_range = 0.1,
        height_shift_range = 0.1,
        zoom_range = 0.1
    )

    return augmenter
