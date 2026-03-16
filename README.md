# Plant Disease Classifier CNN

## Overview
This project implements a custom Convolutional Neural Network (CNN) from scratch using PyTorch to classify agricultural diseases. The model is designed to analyze images of plant leaves and accurately categorize them into three specific health states: Healthy, Powdery Mildew, and Rust. 

The core of this project is a dynamic, object-oriented PyTorch model class that allows for highly customizable architecture generation, bridging the gap between PyTorch's tensor operations and Scikit-learn's user-friendly `.fit()` and `.predict()` pipeline.

## Key Features
* **Custom VGG-Style Architecture:** A deep learning model built from the ground up, utilizing sequential convolutional blocks and max pooling to extract spatial features, followed by a multi-layer perceptron (MLP) classification head.
* **Dynamic Layer Generation:** The `cnn_module` class allows users to dynamically adjust the number of convolutional layers, feature maps, and MLP hidden layers simply by passing arguments during instantiation.
* **Scikit-Learn Style API:** Integrated custom `.fit()` and `.predict()` methods for clean and intuitive training loops, abstracting away the boilerplate PyTorch training mechanics.
* **Automatic Shape Calculation:** Implements a dummy-tensor forward pass within the initialization block to automatically calculate the flattened input size for the fully connected layers, preventing dimensional mismatch errors.
* **Device Agnostic:** The code automatically detects and routes tensors and model weights to a CUDA-enabled GPU if available, drastically reducing training time.

## Dataset and Preprocessing
The model is trained on an image dataset organized into three classes:
1. Healthy
2. Powdery
3. Rust

Images are loaded using `torchvision.datasets.ImageFolder` and processed through a robust data augmentation and normalization pipeline to prevent overfitting:
* Resized to 128x128 pixels.
* Converted to PyTorch Tensors.
* Normalized using standard RGB mean and standard deviation values.
* (Optional) Random horizontal flips and rotations for artificial data expansion.

## Model Performance
After optimizing the architecture to utilize 3 convolutional blocks (32 feature maps each) and 3 hidden MLP layers (256 neurons each) alongside the Adam optimizer and ReLU activations, the model successfully overcame initial mode collapse and vanishing gradient challenges.

* **Testing Accuracy:** Achieved 87.5% accuracy on unseen test data.
* **Training Convergence:** The loss curve demonstrates smooth and rapid convergence without severe overfitting.

## Technologies Used
* **Python 3**
* **PyTorch** (Core model architecture, autograd, and optimization)
* **Torchvision** (Data loading and image transformations)
* **Matplotlib** (Data visualization and loss curve plotting)
* **tqdm** (Training progress tracking)
* **Scikit-learn & Seaborn** (Confusion matrix generation and evaluation)