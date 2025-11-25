import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns

def plot_accuracy_comparison(history_no_aug, history_aug, save_path="../results/plots/accuracy_comparison.png"):
    plt.figure(figsize=(10,4))
    plt.plot(history_no_aug.history['val_accuracy'], label='Without Augmentation')
    plt.plot(history_aug.history['val_accuracy'], label='With Augmentation')
    plt.title("Validation Accuracy Comparison")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    plt.savefig(save_path)
    plt.show()


def plot_loss_comparison(history_no_aug, history_aug, save_path="../results/plots/loss_comparison.png"):
    plt.figure(figsize=(10,4))
    plt.plot(history_no_aug.history['val_loss'], label='Without Augmentation')
    plt.plot(history_aug.history['val_loss'], label='With Augmentation')
    plt.title("Validation Loss Comparison")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    plt.savefig(save_path)
    plt.show()

def plot_confusion_matrix_single(model, x_test, y_test, title="Confusion Matrix", save_path="../results/plots/confusion_matrix.png"):

    y_true = np.argmax(y_test, axis=1)
    y_pred = np.argmax(model.predict(x_test), axis=1)

    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(7,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap="Blues")
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.show()

def plot_confusion_matrices_side_by_side(model_no_aug, model_aug, x_test, y_test, save_path="../results/plots/confusion_matrix_comparison.png"):

    y_true = np.argmax(y_test, axis=1)

    y_pred_no_aug = np.argmax(model_no_aug.predict(x_test), axis=1)
    y_pred_aug = np.argmax(model_aug.predict(x_test), axis=1)

    cm_no_aug = confusion_matrix(y_true, y_pred_no_aug)
    cm_aug = confusion_matrix(y_true, y_pred_aug)

    fig, axs = plt.subplots(1, 2, figsize=(14,5))

    sns.heatmap(cm_no_aug, annot=True, fmt='d', cmap="Blues", ax=axs[0])
    axs[0].set_title("No Augmentation")
    axs[0].set_xlabel("Predicted")
    axs[0].set_ylabel("True")

    sns.heatmap(cm_aug, annot=True, fmt='d', cmap="Greens", ax=axs[1])
    axs[1].set_title("With Augmentation")
    axs[1].set_xlabel("Predicted")
    axs[1].set_ylabel("True")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.show()

def show_example_predictions(model_no_aug, model_aug, x_test, y_test, n=5):
    import numpy as np
    import matplotlib.pyplot as plt

    for i in range(n):
        idx = np.random.randint(0, len(x_test))
        img = x_test[idx]

        pred_no_aug = model_no_aug.predict(img.reshape(1,28,28,1)).argmax()
        pred_aug = model_aug.predict(img.reshape(1,28,28,1)).argmax()
        true = y_test[idx].argmax()

        plt.figure(figsize=(5,5))
        plt.imshow(img.reshape(28,28), cmap="gray")
        plt.title(
            f"True: {true}\n"
            f"No Aug: {pred_no_aug}   |   With Aug: {pred_aug}"
        )
        plt.axis("off")
        plt.show()

