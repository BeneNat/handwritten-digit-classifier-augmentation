import os
from .data import load_data
from .augment import get_augmenter
from .model import build_cnn

def train(epochs=5, batch_size=32, save_model=True, model_path='../results/models/cnn_augmented.keras'):
    x_train, y_train, x_test, y_test = load_data()

    augmenter = get_augmenter()
    augmenter.fit(x_train)

    model = build_cnn()

    history = model.fit(
        augmenter.flow(x_train, y_train, batch_size=batch_size),
        validation_data=(x_test, y_test),
        epochs=epochs
    )

    if save_model:
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        model.save(model_path)
        print(f"Model saved to: {model_path}")

    return history, model

def train_no_augmentation(epochs=5, batch_size=32, save_model=True, model_path='../results/models/cnn_normal.keras'):
    x_train, y_train, x_test, y_test = load_data()

    model = build_cnn()

    history = model.fit(
        x_train, y_train,
        validation_data=(x_test, y_test),
        epochs=epochs,
        batch_size=batch_size
    )

    if save_model:
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        model.save(model_path)
        print(f"Model saved to: {model_path}")

    return history, model
