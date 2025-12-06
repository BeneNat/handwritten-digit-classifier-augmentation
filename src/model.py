from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense

def build_cnn():
    """
    Defines the CNN architecture used for both experiments.
    """
    model = Sequential([
        # First convolutional block
        Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        MaxPool2D(2, 2),

        # Second convolutional block
        Conv2D(64, (3, 3), activation='relu'),

        # Flatten and dense layers for classification
        Flatten(),
        Dense(128, activation='relu'),
        Dense(10, activation='softmax')
    ])

    # Compile with Adam optimizer and categorical crossentropy loss
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model
