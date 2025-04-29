# Binary Classification CNN 

This project demonstrates the implementation of a Convolutional Neural Network (CNN) for binary classification using the MNIST dataset. The CNN model is built using the Keras framework in Python.

## Dataset
The project uses a partial MNIST dataset, consisting of images of handwritten digits. The dataset is stored in two files: `training.npz` for training data and `test.npz` for testing data. The labels are binary, indicating whether the digit is a specific class or not.

## Model Architecture
The CNN model architecture consists of the following layers:
- Two convolutional layers with 32 and 64 filters, respectively, and ReLU activation.
- Max pooling layers after each convolutional layer.
- Flatten layer to convert the 2D feature maps into a 1D feature vector.
- Fully connected layer with 64 neurons and ReLU activation.
- Dropout layer with a rate of 0.5 to prevent overfitting.
- Output layer with 2 neurons and softmax activation for binary classification.

## Training and Evaluation
The model is trained using the binary cross-entropy loss function and the Adam optimizer. The training is performed for 15 epochs with a batch size of 128. The model's performance is evaluated on the test set, achieving an accuracy of approximately 98.5%.

## Requirements
- Python 3.x
- NumPy
- Keras

## Usage
1. Clone the repository.
2. Install the required dependencies.
3. Run the `perceptron.py` script to train and evaluate the CNN model.

## Results
The trained model achieves high accuracy on both the training and test sets, demonstrating its effectiveness in binary classification of handwritten digits.
