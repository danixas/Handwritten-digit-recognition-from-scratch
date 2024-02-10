# Neural Network for handwritten digit recognition from scratch

## Overview
The project takes MNIST dataset of handwritten digits, implements a neural network with all the necessary algorithms like feedforward and backpropagation (without the use of libraries like TensorFlow or PyTorch), and then trains the model on the MNIST data
- You can make your own configurations by changing them in the constants.py file, and then train your own model
- You can run the test.py which will feed test data through a saved model (default is: saved_nn_reworked89.38.pkl)
with the current configuration the model achieves 89.38% accuracy on test data which it has never seen before

## Getting Started
# Clone the repository
git clone https://github.com/your-username/your-project.git
# Navigate to the project directory

```bash
cd Handwritten-digit-recognition-from-scratch
```

# Prerequisites

Before you start, make sure you have Python and `virtualenv` installed. If not, you can download Python from [python.org](https://www.python.org/downloads/) and install `virtualenv` using:
```bash
pip install virtualenv
```
# Create a virtual environment
Create a virtual environment to isolate your project dependencies. Open a terminal in your project directory and run:
```bash
python -m venv venv
```
# Activate the virtual environment
# On Windows
```bash
venv\Scripts\activate
```
# On macOS and Linux
```bash
source venv/bin/activate
```

# Install Dependencies
With the virtual environment activated, install the project dependencies from the requirements.txt file:

```bash
pip install -r requirements.txt
```
now you are ready to run the project


# Usage 1 (Train your own model)
To Train your own model, you can configure the constants.py (optional) and the run:
```bash
python train.py
```
it will train a new model and then save it to .pkl file which you can then specify to test in the test.py file
# Usage 2 (Test a model)
To Test a model first you can either just run the test.py, it will test the already added and trained model in the ```bash
python train.py
```bash
python test.py
```

## Acknowledgments

A part of the math that was used to code the algorithms for backpropogation and other parts was inspired from a youtube channel 3Blue1Brown. (https://www.youtube.com/watch?v=aircAruvnKk&list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi)

---
The essence of the project is to understand how the neural networks actually work behind the scenes, so that when you use the more optimized options like tensorflow to build your models, you can actually understand and come up with more educated solutions to problems.
