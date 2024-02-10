import numpy
import pickle
import matplotlib.pyplot as plt
from MNIST import train_images, train_labels, test_images, test_labels


def get_modified_data(image, i, labels):
    modified_training = image.flatten()
    modified_training = modified_training / 255
    outputs = numpy.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    outputs[labels[i]] = 1
    return modified_training, outputs


def test():
    with open('saved_nn_reworked89.38.pkl', 'rb') as INPUT:
        nn = pickle.load(INPUT)

    accurate = 0
    total = 0
    for i, image in enumerate(test_images):
        plt.imshow(image/255, cmap="gray")
        modified_test, outputs = get_modified_data(image, i, test_labels)
        nn.modify_input(modified_test, outputs)
        nn.feed_forward()
        if nn.correct_number_picked():
            accurate += 1
        total += 1

    print("test accuracy: ", round(accurate / total * 100, 2), "%")


test()
