import numpy as np
import math
from random import random
from constants import BATCH_SIZE


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


vectorized_sigmoid = np.vectorize(sigmoid)


def d_sigmoid(x):
    return 1 / (1 + math.exp(-x)) * (1 - 1 / (1 + math.exp(-x)))


vectorized_d_sigmoid = np.vectorize(d_sigmoid)


def random_nr():
    return -1 + random() * 2


class NeuralNetwork:
    def __init__(self, inputs, output, h_count, h_sizes):
        self.h_count = h_count
        self.h_sizes = h_sizes
        self.layer_count = h_count + 2

        self.inputs = inputs
        self.output_expected = output

        if self.inputs is not None and self.output_expected is not None:
            self.i_count = len(inputs)
            self.o_count = len(output)
            # layer initialization (holds layer activations)

            self.layers = [[] for _ in range(self.layer_count)]
            self.layers[0] = self.inputs
            x = 5
            for i in range(h_count):
                self.layers[i + 1] = np.array(0 for _ in range(h_sizes[i]))
            self.layers[-1] = [0 for _ in range(self.o_count)]

            # weighted sums initialization (same as layer just not passed through sigmoid function)
            self.weighted_sums = [[] for _ in range(self.layer_count)]
            self.weighted_sums[0] = inputs
            for i in range(h_count):
                self.weighted_sums[i + 1] = [0 for _ in range(h_sizes[i])]
            self.weighted_sums[-1] = [0 for _ in range(self.o_count)]

            # error initialization
            self.errors = [[] for _ in range(self.layer_count)]
            self.errors[0] = inputs
            for i in range(h_count):
                self.errors[i + 1] = [0 for _ in range(h_sizes[i])]
            self.errors[-1] = [0 for _ in range(self.o_count)]

            # overall error initialization (the cost of the network that adds up over a set(BATCH_SIZE) of examples)
            # self.overall_errors = np.array([] for _ in range(self.layer_count))
            # self.overall_errors[0] = inputs
            # for i in range(h_count):
            #     self.overall_errors[i + 1] = np.array(0 for _ in range(h_sizes[i]))
            # self.overall_errors[-1] = np.array(0 for _ in range(self.o_count))

            # expected layer value initialization (holds layer activations)
            # self.layers_expected = np.array([] for _ in range(self.layer_count))
            # self.layers_expected[0] = inputs
            # for i in range(h_count):
            #     self.layers_expected[i + 1] = np.array(0 for _ in range(h_sizes[i]))
            # self.layers_expected[-1] = np.array(0 for _ in range(self.o_count))

            # weight initialization (setting to random number between -1 and 1)
            self.weights = [[[]] for _ in range(h_count + 1)]
            self.weights[0] = [[random_nr() for _ in range(h_sizes[0])] for _ in range(self.i_count)] # [784 x h_size] input to first hidden layer weights
            for i in range(1, h_count):
                self.weights[i] = [[random_nr() for _ in range(h_sizes[i - 1])] for _ in range(h_sizes[i])]
            self.weights[h_count] = [[random_nr() for _ in range(self.o_count)] for _ in range(h_sizes[-1])] # [h_size x 10] last hidden to output layer weights

            # weight change initialization (setting to 0)
            self.weights_changes = [[[]] for _ in range(h_count + 1)]
            self.weights_changes[0] = [[0 for _ in range(h_sizes[0])] for _ in range(self.i_count)]  # [784 x h_size] input to first hidden layer weight changes
            for i in range(1, h_count):
                self.weights_changes[i] = [[0 for _ in range(h_sizes[i - 1])] for _ in range(h_sizes[i])]
            self.weights_changes[h_count] = [[0 for _ in range(self.o_count)] for _ in range(h_sizes[-1])]  # [h_size x 10] last hidden to output layer weight changes

    # fixed
    def modify_input(self, inputs, outputs):
        self.layers[0] = inputs
        self.inputs = inputs
        self.output_expected = outputs

    # fixed
    def modify_weights(self):
        for i in range(self.layer_count - 1):
            # adding weight changes
            self.weights[i] = (np.add(np.asarray(self.weights[i]), np.asarray(self.weights_changes[i]))).tolist()
            # resetting weight changes
            if i == 0:
                self.weights_changes[i] = [[0 for _ in range(self.h_sizes[0])] for _ in range(self.i_count)]
            elif i == self.layer_count - 2:
                self.weights_changes[i] = [[0 for _ in range(self.o_count)] for _ in range(self.h_sizes[-1])]
            else:
                self.weights_changes[i] = [[0 for _ in range(self.h_sizes[i - 1])] for _ in range(self.h_sizes[i])]

    def feed_forward(self):
        for i in range(1, self.layer_count):
            self.weighted_sums[i] = np.dot(np.transpose(np.asarray(self.weights[i - 1])), np.asarray(self.layers[i - 1])).tolist()
            self.layers[i] = vectorized_sigmoid(self.weighted_sums[i])

    # Calculate the derivative of an neuron output
    def transfer_derivative(self, output):
        return output * (1.0 - output)

    def back_propagation(self):
        for l in range(self.layer_count - 1, -1, -1):
            layer = self.layers[l]
            if l != self.layer_count - 1:
                # current layer
                for j in range(len(layer)):
                    error = 0.0
                    # layer to the right
                    for k in range(len(self.layers[l + 1])):
                        # error += (self.weights[l][j][k] * self.errors[l + 1][k] * self.transfer_derivative(self.layers[l + 1][k]))
                        error += 2 * self.errors[l + 1][k] * d_sigmoid(self.weighted_sums[l + 1][k]) * self.weights[l][j][k]
                    self.errors[l][j] = error

            else:
                self.errors[l] = (np.subtract(self.output_expected, self.layers[l])).tolist()
                # self.overall_errors[l] = (np.add(self.overall_errors[l], self.errors[l] / BATCH_SIZE)).tolist()
        self.calculate_weight_changes()

    def calculate_weight_changes(self):
        for l in range(1, self.layer_count):
            for i in range(len(self.weights_changes[l - 1])):
                for j in range(len(self.weights_changes[l - 1][i])):
                    self.weights_changes[l - 1][i][j] = self.weights_changes[l - 1][i][j] + (2 * (self.errors[l][j]) * d_sigmoid(self.weighted_sums[l][j]) * self.layers[l - 1][i]) / BATCH_SIZE

    def correct_number_picked(self):
        expected = np.where(self.output_expected == np.amax(self.output_expected))
        found = np.where(self.layers[-1] == np.amax(self.layers[-1]))
        return expected == found

    def number_picked(self):
        found = np.where(self.layers[-1] == np.amax(self.layers[-1]))
        return found
