import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

"""
Evaluation utilities for MNIST models.
Includes:
- Cleaner plotting
- Automatic figure titles
- Optional saving
- Additional helper functions: classification report, sample grid
- Consistent styling
"""

# ---------------------------------------------------------
# ensure save directory exists
# ---------------------------------------------------------
def _ensure_dir(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)

# ---------------------------------------------------------
# Accuracy comparison
# ---------------------------------------------------------
def plot_accuracy_comparison(history_no_aug, history_aug, save_path="../results/plots/accuracy_comparison.png"):
    plt.figure(figsize=(10, 5))
    plt.plot(history_no_aug.history['val_accuracy'], label='Without Augmentation', linewidth=2)
    plt.plot(history_aug.history['val_accuracy'], label='With Augmentation', linewidth=2)

    plt.title("Validation Accuracy Comparison", fontsize=14)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Accuracy", fontsize=12)
    plt.grid(alpha=0.3)
    plt.legend()

    _ensure_dir(save_path)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()

# ---------------------------------------------------------
# Loss comparison
# ---------------------------------------------------------
def plot_loss_comparison(history_no_aug, history_aug, save_path="../results/plots/loss_comparison.png"):
    plt.figure(figsize=(10, 5))
    plt.plot(history_no_aug.history['val_loss'], label='Without Augmentation', linewidth=2)
    plt.plot(history_aug.history['val_loss'], label='With Augmentation', linewidth=2)

    plt.title("Validation Loss Comparison", fontsize=14)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.grid(alpha=0.3)
    plt.legend()

    _ensure_dir(save_path)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()

# ---------------------------------------------------------
# Single confusion matrix
# ---------------------------------------------------------
def plot_confusion_matrix_single(model, x_test, y_test,
                                 title="Confusion Matrix",
                                 save_path="../results/plots/confusion_matrix.png"):
    y_true = np.argmax(y_test, axis=1)
    y_pred = np.argmax(model.predict(x_test), axis=1)

    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", cbar=False)
    plt.title(title, fontsize=14)
    plt.xlabel("Predicted", fontsize=12)
    plt.ylabel("True", fontsize=12)

    _ensure_dir(save_path)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()

# ---------------------------------------------------------
# Side-by-side confusion matrices
# ---------------------------------------------------------
def plot_confusion_matrices_side_by_side(model_no_aug, model_aug, x_test, y_test,
                                         save_path="../results/plots/confusion_matrix_comparison.png"):
    y_true = np.argmax(y_test, axis=1)
    y_pred_no_aug = np.argmax(model_no_aug.predict(x_test), axis=1)
    y_pred_aug = np.argmax(model_aug.predict(x_test), axis=1)

    cm_no_aug = confusion_matrix(y_true, y_pred_no_aug)
    cm_aug = confusion_matrix(y_true, y_pred_aug)

    fig, axs = plt.subplots(1, 2, figsize=(14, 5))

    sns.heatmap(cm_no_aug, annot=True, fmt='d', cmap="Blues", ax=axs[0], cbar=False)
    axs[0].set_title("No Augmentation", fontsize=14)
    axs[0].set_xlabel("Predicted")
    axs[0].set_ylabel("True")

    sns.heatmap(cm_aug, annot=True, fmt='d', cmap="Greens", ax=axs[1], cbar=False)
    axs[1].set_title("With Augmentation", fontsize=14)
    axs[1].set_xlabel("Predicted")
    axs[1].set_ylabel("True")

    _ensure_dir(save_path)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()

# ---------------------------------------------------------
# Example predictions
# ---------------------------------------------------------
def show_example_predictions(model_no_aug, model_aug, x_test, y_test, n=5):
    for _ in range(n):
        idx = np.random.randint(0, len(x_test))
        img = x_test[idx]

        pred_no_aug = model_no_aug.predict(img.reshape(1, 28, 28, 1), verbose=0).argmax()
        pred_aug = model_aug.predict(img.reshape(1, 28, 28, 1), verbose=0).argmax()
        true = y_test[idx].argmax()

        plt.figure(figsize=(5, 5))
        plt.imshow(img.reshape(28, 28), cmap="gray")
        plt.title(f"True: {true}\nNo Aug: {pred_no_aug} | With Aug: {pred_aug}", fontsize=13)
        plt.axis("off")
        plt.show()

# ---------------------------------------------------------
# Classification report helper
# ---------------------------------------------------------
def print_classification_report(model, x_test, y_test):
    y_true = np.argmax(y_test, axis=1)
    y_pred = np.argmax(model.predict(x_test), axis=1)

    print("\nClassification Report:\n")
    print(classification_report(y_true, y_pred))

# ---------------------------------------------------------
# Image grid helper
# ---------------------------------------------------------
def show_image_grid(images, labels=None, rows=4, cols=4):
    plt.figure(figsize=(cols * 2, rows * 2))
    for i in range(rows * cols):
        if i >= len(images):
            break
        plt.subplot(rows, cols, i + 1)
        plt.imshow(images[i].reshape(28, 28), cmap="gray")
        title = f"Label: {labels[i]}" if labels is not None else ""
        plt.title(title, fontsize=9)
        plt.axis("off")
    plt.tight_layout()
    plt.show()
