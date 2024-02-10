import numpy
from random import seed
from constants import*
from NN import NeuralNetwork
from MNIST import train_images, train_labels, test_images, test_labels
import pickle

numpy.set_printoptions(precision=3)
seed(2)


def save_nn(filename, nn):
    with open(filename, 'wb') as output:
        pickle.dump(nn, output, pickle.HIGHEST_PROTOCOL)


def get_modified_data(image, i, labels):
    modified_training = image.flatten()
    modified_training = modified_training / 255
    outputs = numpy.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    outputs[labels[i]] = 1
    return modified_training, outputs


def main():
    total = 0
    accurate = 0
    init_index = 0
    init_image = train_images[init_index]
    init_modified_training, init_outputs = get_modified_data(init_image, init_index, train_labels)
    network = NeuralNetwork(init_modified_training, init_outputs, HIDDEN_LAYER_COUNT, HIDDEN_LAYER_SIZES)
    print("training data size: ", len(train_images) )
    print("BATCH SIZE: ", BATCH_SIZE)
    for _ in range(EPOCHS):
        for i, image in enumerate(train_images):
            modified_training, outputs = get_modified_data(image, i, train_labels)
            if i >= TRAIN_SIZE:
                break
            elif i % BATCH_SIZE == 0:
                network.modify_input(modified_training, outputs)
                network.modify_weights()
                continue

            network.modify_input(modified_training, outputs)
            network.feed_forward()
            network.back_propagation()

            percentage_done = round(i/TRAIN_SIZE*100, 2)
            print("Network Training: ", percentage_done, "%", "  done")

    for i, image in enumerate(test_images):
        modified_test, outputs = get_modified_data(image, i, test_labels)
        network.modify_input(modified_test, outputs)
        network.feed_forward()
        if network.correct_number_picked():
            accurate += 1
        total += 1

    print("test accuracy: ", accurate / total * 100, "%")
    name = "saved_nn_reworked" + str(round(accurate / total * 100, 2)) + '.pkl'
    save_nn(name, network)

main()
