from pure_python.neuron import Neuron
from math import pow


class OneLayerNet:

    def __init__(self, inputs_count, output_neurons_count):
        self.__inputs_count = inputs_count
        self.__neurons = []
        self.__eta = 0.5
        for j in range(output_neurons_count):
            self.__neurons.append(Neuron(inputs_count))

    def train_delta_rule(self, vector):
        for neuron in self.__neurons:
            neuron.delta_calc_y(vector.get_x())
        errors = [0] * len(self.__neurons)
        error = 0.0

        for i, neuron in enumerate(self.__neurons):
            errors[i] = vector.get_d()[i] - neuron.get_y()
            error += errors[i]

        for i, neuron in enumerate(self.__neurons):
            neuron_weights = neuron.get_weights()
            delta_weights = [0] * (len(neuron_weights) + 1)

            for j, weight in enumerate(delta_weights):
                delta_weights[j] += self.__eta * errors[i] * vector.get_x()[j]
            neuron.correct_weights(delta_weights)

        return error

    def train(self, vector, learning_rate):

        for j in range(len(self.__neurons)):
            self.__neurons[j].calc_y(vector.get_x())

        weights_deltas = [[0] * (len(vector.get_x()) + 1)] * len(self.__neurons)

        for j in range(len(self.__neurons)):
            sigma = (vector.get_d()[j] - self.__neurons[j].get_y()) \
                    * self.__neurons[j].derivative()
            weights_deltas[j][0] = learning_rate * sigma
            wlen = len(self.__neurons[j].get_weights())
            for i in range(wlen):
                weights_deltas[j][i] = learning_rate * sigma * vector.get_x()[i]
            self.__neurons[j].correct_weights(weights_deltas[j])

        loss = 0
        for j in range(len(self.__neurons)):
            loss += pow(vector.get_d()[j] - self.__neurons[j].get_y(), 2)

        return 0.5 * loss

    def test(self, vector):
        y = [0] * len(self.__neurons)
        for j in range(len(self.__neurons)):
            self.__neurons[j].calc_y(vector.get_x())
            y[j] = self.__neurons[j].get_y()
        return y




















