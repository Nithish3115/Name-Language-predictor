# 🌐 Detecting the Language of a Person’s Name using a PyTorch RNN

## Introduction

A Recurrent Neural Network (RNN) in PyTorch that will classify people’s names by their languages. 

## Table of Contents

- [Introduction](#introduction)
- [Dependencies](#dependencies)
- [Data Pre-processing](#data-pre-processing)
- [Turning the Names into PyTorch Tensors](#turning-the-names-into-pytorch-tensors)
- [Building the RNN](#building-the-rnn)
- [Testing the RNN](#testing-the-rnn)
- [Training the RNN](#training-the-rnn)
- [Plotting the Results](#plotting-the-results)
- [Evaluating the Results](#evaluating-the-results)
- [Predicting New Names](#predicting-new-names)
- [Conclusion](#conclusion)
- [References](#references)

## Dependencies

Before starting, ensure you have the following dependencies installed:

- Python 3.6+
- PyTorch
- NumPy
- Matplotlib

You can install the necessary Python packages using pip:

```sh
pip install torch numpy matplotlib
```
# Data Pre-processing
 
 ### 📂 Loading Data
We use the glob module to load all text files in the dataset.

 ### 🔄 Converting Unicode to ASCII
To handle names in Unicode format, we convert them to ASCII to remove diacritics.

 ### 🗂 Creating Language Dictionaries
We create a dictionary with a list of names for each language and count the number of languages in the dataset.

 ### Turning the Names into PyTorch Tensors
When working with data in PyTorch, we need to convert it to PyTorch tensors. Each letter in a name is converted into a one-hot vector, and a name is represented as a sequence of these vectors.

### 🏗 Building the RNN
We define an RNN class using torch.nn.Module, which includes an initialization method, a forward method for processing inputs, and a method to initialize hidden states.

### 🧪 Testing the RNN
We create an instance of the RNN class and test it with sample inputs to ensure it produces outputs of the correct size.

 ###  🏋️ Training the RNN
🎲 Preparing Training Data
We define a function to obtain a random training example and its corresponding tensor representation.

### 📉 Defining Loss Function and Optimizer
We use negative log likelihood loss and stochastic gradient descent for training the model.

### 🔄 Training Loop
We run the training process for a specified number of epochs, keeping track of the loss and printing the model’s progress.

### 📊 Plotting the Results
Using Matplotlib, we plot the loss over time to visualize the training process and the model's learning rate.
##
### 📈 Evaluating the Results
We create a confusion matrix to evaluate the model's performance on different categories. This helps us understand which languages are often misclassified.

### 🔍 Predicting New Names
We define a function to predict the likely languages of new names, displaying the top predictions with their probabilities.

### 🎉 Conclusion
This is build and train a simple RNN in PyTorch to classify names by language.  

### 📚 References
PyTorch Documentation

Tutorials on RNNs by Brian Mwangi

Related articles on machine learning and neural networks
