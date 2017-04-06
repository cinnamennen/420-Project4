import numpy as np
import plotly.plotly as py
import plotly.graph_objs as go


def random_float_range(low, high):
    return (high - low) * np.random.random_sample() + low


def next_thing():
    while True:
        yield random_float_range(-.1, .1)


class Train:
    def __init__(self, hw, inputs, hb):
        """

        :type hw: list[list[list[float]]]
        :type hb: list[list[float]]
        """
        self.bias = hb
        self.weights = hw

        self.output_delta = 0
        self.output_sigma = 0
        self.output_h = 0

        self.number_of_inputs = inputs

        self.h = None
        self.sigma = None
        self.delta = None

        self.r = []
        self.best_root_mean_square_error = 2 ** 64
        self.best_weights = None

        self.reset()

    def reset(self):
        self.h = [[0 for _ in layer] for layer in self.weights]  # layer - neuron
        self.sigma = [[0 for _ in layer] for layer in self.weights]  # layer - neuron
        self.delta = [[0 for _ in layer] for layer in self.weights]  # layer - neuron

    def train_net(self, training_patterns, learning_rate):
        for pattern in training_patterns:
            self.reset()
            self.compute_outputs(pattern)
            self.compute_deltas(pattern[-1])
            self.update_weights(learning_rate, pattern)

    def compute_deltas(self, expected_output):
        for layer_index, layer in reversed(list(enumerate(self.weights))):
            for neuron_index, neuron in enumerate(layer):
                if layer_index == len(self.weights) - 1:
                    self.delta[layer_index][neuron_index] = self.sigma[layer_index][neuron_index] * \
                                                            (1 - self.sigma[layer_index][neuron_index]) * \
                                                            (expected_output
                                                             - self.sigma[layer_index][neuron_index])
                else:
                    for next_neuron_index, next_neuron in enumerate(self.weights[layer_index + 1]):
                        self.delta[layer_index][neuron_index] += \
                            self.sigma[layer_index][neuron_index] * \
                            (1 - self.sigma[layer_index][neuron_index]) * \
                            self.delta[layer_index + 1][next_neuron_index] * \
                            self.weights[layer_index + 1][next_neuron_index][neuron_index]

    def compute_outputs(self, training_pattern):
        for layer_index, layer in enumerate(self.weights):  # layer - neuron - weight
            for neuron_index, neuron in enumerate(layer):  # neuron - weight
                if layer_index == 0:
                    for i in range(self.number_of_inputs):
                        self.h[layer_index][neuron_index] += \
                            self.weights[layer_index][neuron_index][i] * \
                            training_pattern[i]
                else:
                    for previous_neuron_index in range(len(self.weights[layer_index - 1])):
                        self.h[layer_index][neuron_index] += \
                            self.weights[layer_index][neuron_index][previous_neuron_index] * \
                            self.sigma[layer_index - 1][previous_neuron_index]

                self.h[layer_index][neuron_index] += self.bias[layer_index][neuron_index]
                self.sigma[layer_index][neuron_index] = 1.0 / (1 + np.exp(-self.h[layer_index][neuron_index]))

    def get_output(self, pattern):
        for layer_index, layer in enumerate(self.weights):  # layer - neuron - weight
            for neuron_index, neuron in enumerate(layer):  # neuron - weight
                if layer_index == 0:
                    for i in range(self.number_of_inputs):
                        self.h[layer_index][neuron_index] += \
                            self.weights[layer_index][neuron_index][i] * \
                            pattern[i]
                else:
                    for previous_neuron_index in range(len(self.weights[layer_index - 1])):
                        self.h[layer_index][neuron_index] += \
                            self.weights[layer_index][neuron_index][previous_neuron_index] * \
                            self.sigma[layer_index - 1][previous_neuron_index]

                self.h[layer_index][neuron_index] += self.bias[layer_index][neuron_index]
                self.sigma[layer_index][neuron_index] = 1.0 / (1 + np.exp(-self.h[layer_index][neuron_index]))
        return self.h[-1][0]

    def update_weights(self, learning_rate, training_pattern):
        for layer_index, layer in enumerate(self.weights):  # layer - neuron - weight
            for neuron_index, neuron in enumerate(layer):  # neuron - weight
                if layer_index == 0:
                    for i in range(self.number_of_inputs):  # weight
                        self.weights[layer_index][neuron_index][i] += \
                            learning_rate * \
                            self.delta[layer_index][neuron_index] \
                            * training_pattern[i]
                else:
                    for previous_neuron_index in range(len(self.weights[layer_index - 1])):  # weight
                        self.weights[layer_index][neuron_index][previous_neuron_index] += learning_rate * \
                                                                                          self.delta[layer_index][
                                                                                              neuron_index] * \
                                                                                          self.sigma[layer_index - 1][
                                                                                              previous_neuron_index]
                    self.bias[layer_index][neuron_index] += learning_rate * self.delta[layer_index][neuron_index]

    def evaluate_net(self, evaluation_tests):
        """

        :rtype: float
        :type evaluation_tests: list[list[float]]
        """
        root_mean_square_error = self.get_root_mean_square_error(evaluation_tests)
        if root_mean_square_error < self.best_root_mean_square_error:
            self.best_weights = self.weights.copy()
        return root_mean_square_error

    def get_root_mean_square_error(self, evaluation_tests):
        """

        :type evaluation_tests: list[list[float]]
        :rtype: float
        """
        total = 0
        for test in evaluation_tests:
            self.reset()
            output = self.get_output(test)
            expected = test[-1]
            total += (expected - output) ** 2
        root_mean_square_error = np.sqrt((1 / (2 * len(evaluation_tests))) * total)
        return root_mean_square_error

    def validate_net(self, validation_patterns):
        self.r.append(self.evaluate_net(validation_patterns))

    def test_net(self, testing_patterns):
        return self.evaluate_net(testing_patterns)


def main(number_of_neurons_in_layer: list,
         learning_rate: float,
         training_file: str,
         validation_file: str,
         testing_file: str,
         number_of_training_epochs: int,
         inputs: int):
    with open(training_file, 'r') as f:
        training = [list(map(float, line.strip().split(" "))) for line in f]

    with open(validation_file, 'r') as f:
        validation = [list(map(float, line.strip().split(" "))) for line in f]

    with open(testing_file, 'r') as f:
        testing = [list(map(float, line.strip().split(" "))) for line in f]

    gen = next_thing()

    number_of_neurons_in_layer += [1]

    # Initialize the weights
    hidden_bias = []
    hidden_weights = []
    for index, layer in enumerate(number_of_neurons_in_layer):
        hidden_layer = []
        bias_layer = []
        for neuron in range(layer):
            hidden_neuron = []
            if index == 0:
                # todo can we remove this special casing?
                for previous in range(inputs):
                    # float_range = random_float_range(-0.1, 0.1)
                    hidden_neuron.append(next(gen))
            else:
                for previous in range(number_of_neurons_in_layer[index - 1]):
                    hidden_neuron.append(next(gen))
            hidden_layer.append(hidden_neuron)
            bias_layer.append(next(gen))

        hidden_weights.append(hidden_layer)
        hidden_bias.append(bias_layer)

    # print(hidden_weights)

    t = Train(hidden_weights, inputs, hidden_bias)

    for i in range(number_of_training_epochs):
        t.train_net(training_patterns=training, learning_rate=learning_rate)  # train the network 1 iteration
        t.validate_net(validation)  # calcluate the RSME for this epoch
    best_root_square_mean_error = t.test_net(testing)  # Analyze the RSME for the best validated net
    print(best_root_square_mean_error)

    pairs = [(x, y,) for x, y in enumerate(t.r)]
    x = [x for (x, y,) in pairs]
    y = [y for (x, y,) in pairs]

    trace = go.Scatter(
        x=x,
        y=y,
        mode='markers'
    )

    data = [trace]
    plot_url = py.plot(data, filename='basic-line')
    print(plot_url)


if __name__ == '__main__':
    hidden = [2, 3, 2]
    main(hidden, 0.1, "training1.txt", "validation1.txt", "testing1.txt", 10000, 2)
