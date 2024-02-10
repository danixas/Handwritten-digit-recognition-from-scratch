# File to download the MNIST dataset without having to import tensorflow.keras

import requests
import gzip
import os
import struct
import numpy as np

# Define the URLs for the MNIST dataset
mnist_urls = {
    'train_images': 'http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz',
    'train_labels': 'http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz',
    'test_images': 'http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz',
    'test_labels': 'http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz'
}

# Specify the folder location for the MNIST dataset
mnist_folder = 'MNIST_data'

# Function to download and extract the dataset
def download_mnist():
    for key, url in mnist_urls.items():
        filename = os.path.join(mnist_folder, url.split("/")[-1])
        if not os.path.exists(filename):
            print(f"Downloading {filename}...")
            response = requests.get(url, stream=True)
            with open(filename, 'wb') as file:
                for chunk in response.iter_content(chunk_size=1024):
                    if chunk:
                        file.write(chunk)
            print(f"{filename} downloaded successfully.")
            # Extract the file if it is in gzip format
            if filename.endswith('.gz'):
                with gzip.open(filename, 'rb') as f_in:
                    with open(filename[:-3], 'wb') as f_out:
                        f_out.write(f_in.read())
                os.remove(filename)

# Function to read the MNIST images and labels
def read_mnist_images(filename):
    with open(filename, 'rb') as file:
        magic, num_items, num_rows, num_cols = struct.unpack(">IIII", file.read(16))
        data = np.frombuffer(file.read(), dtype=np.uint8).reshape(num_items, num_rows, num_cols)
    return data

def read_mnist_labels(filename):
    with open(filename, 'rb') as file:
        magic, num_items = struct.unpack(">II", file.read(8))
        data = np.frombuffer(file.read(), dtype=np.uint8)
    return data

# Create the MNIST_data folder if it doesn't exist
if not os.path.exists(mnist_folder):
    os.makedirs(mnist_folder)

# Download and extract the MNIST dataset
download_mnist()

# Read the training images and labels
train_images = read_mnist_images(os.path.join(mnist_folder, 'train-images-idx3-ubyte'))
train_labels = read_mnist_labels(os.path.join(mnist_folder, 'train-labels-idx1-ubyte'))

# Read the test images and labels
test_images = read_mnist_images(os.path.join(mnist_folder, 't10k-images-idx3-ubyte'))
test_labels = read_mnist_labels(os.path.join(mnist_folder, 't10k-labels-idx1-ubyte'))

