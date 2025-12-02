# Handwritten Digit Classifier: Data Augmentation Analysis

This project investigates the impact of data augmentation techniques on the performance and generalization capabilities of Convolutional Neural Networks (CNN) applied to the MNIST handwritten digit dataset. It was developed as part of the "Digital Systems Design using High-Level Languages (ESL)" course at AGH University of Krakow.

## Project Overview

The primary objective is to compare two training scenarios using an identical CNN architecture:
1.  **Baseline:** Training on the standard, static MNIST training set.
2.  **Augmented:** Training with real-time data augmentation to expand the diversity of the training set.

The project implements a modular pipeline for data loading, model definition, training, and evaluation, culminating in a comparative analysis of accuracy, loss, and confusion matrices.

## Methodology

### Model Architecture
The classifier is a sequential Convolutional Neural Network built with TensorFlow/Keras, consisting of the following layers:
* Input Layer (28x28x1 grayscale images)
* Conv2D (32 filters, 3x3 kernel, ReLU activation)
* MaxPooling2D (2x2 pool size)
* Conv2D (64 filters, 3x3 kernel, ReLU activation)
* Flatten
* Dense (128 units, ReLU activation)
* Output Dense (10 units, Softmax activation)

Optimizer: Adam
Loss Function: Categorical Crossentropy

### Data Augmentation Strategy
To improve model robustness, the following geometric transformations are applied dynamically during training:
* **Rotation:** ±10 degrees
* **Width Shift:** ±10%
* **Height Shift:** ±10%
* **Zoom:** ±10%

## Project Structure

The codebase is organized into a modular structure to separate configuration, logic, and execution.

```text
├── notebooks/
│   └── main_experiment.ipynb    # Main entry point: runs training and visualization
├── results/
│   ├── models/                  # Directory for saved Keras models
│   └── plots/                   # Generated accuracy/loss plots and confusion matrices
├── src/
│   ├── augment.py               # Configuration of ImageDataGenerator
│   ├── data.py                  # MNIST data loading and preprocessing routines
│   ├── evaluate.py              # Visualization utilities (plots, confusion matrices)
│   ├── model.py                 # CNN architecture definition
│   └── train.py                 # Training loops for both experimental scenarios
├── requirements.txt             # Python dependencies
└── README.md                    # Project documentation
````

## Setup and Installation

To set up the development environment, ensure you have Python 3.8+ installed.

1.  Clone the repository to your local machine:

    ```bash
    git clone https://github.com/BeneNat/handwritten-digit-classifier-augmentation
    cd handwritten-digit-classifier-augmentation
    ```

2.  Create and activate a virtual environment (optional but recommended):

    ```bash
    python -m venv venv
    # Windows:
    .\venv\Scripts\activate
    # Linux/Mac:
    source venv/bin/activate
    ```

3.  Install the required dependencies:

    ```bash
    pip install -r requirements.txt
    ```

## Running the Experiment

The entire experimental pipeline is controlled via the Jupyter Notebook.

1.  Launch the notebook server:

    ```bash
    jupyter notebook notebooks/main_experiment.ipynb
    ```

2.  Execute the cells in the notebook sequentially to perform the following steps:

      * **Data Loading:** Automatically downloads and normalizes the MNIST dataset.
      * **Training:**
          * Trains the baseline model (No Augmentation).
          * Trains the robust model (With Augmentation).
      * **Evaluation:** Calculates metrics and generates comparative visualizations.

## Results

Upon successful execution, the following artifacts will be generated in the `results/` directory:

  * **Trained Models:**

      * `results/models/cnn_normal.keras`: Baseline model weights.
      * `results/models/cnn_augmented.keras`: Augmented model weights.

  * **Visualizations (results/plots/):**

      * `accuracy_comparison.png`: Comparison of validation accuracy over epochs.
      * `loss_comparison.png`: Comparison of validation loss convergence.
      * `confusion_matrix_comparison.png`: Side-by-side heatmap comparison of misclassifications for both models on the test set.

## Project Documentation

The complete project presentation (in Polish) used for the initial proposal is available here:
[Prezentacja Projektu ESL (PDF)](presentation/Prezentacja_Projekt_ESL.pdf)

## Authors and Context

* **Authors:**
    * Filip Żurek
    * Jan Ber
    * Jakub Brachowicz
* **Institution:** AGH University of Krakow
* **Faculty:** Faculty of Computer Science, Electronics and Telecommunications
* **Field of Study:** Electronics and Telecommunications
* **Course:** Digital Systems Design using High-Level Languages (ESL)

## License

This software is distributed under the MIT License. Refer to the [LICENSE](LICENSE) file for the full text.

---
*AGH University of Krakow - ESL Course Project 2025*