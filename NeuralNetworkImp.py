import numpy;


def activation_function(x):
    # Sigmoid function
    return 1 / (1 + numpy.exp(-x))


def gradient_function(x):
    # Sigmoid derivative function
    return x*(1 - x)

def get_random_weight(x, y):
    return numpy.random.rand(x, y)


class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.param_input_size = input_size
        self.param_hidden_size = hidden_size
        self.param_output_size = output_size
        self.hidden_weight = get_random_weight(input_size, hidden_size)
        self.output_weight = get_random_weight(hidden_size, output_size)
        self.forward_propagated = 0
        self.z_hidden = 0
        self.z_input = 0
        self.z_output = 0
        self.propagation_error = 0
        self.propagation_difference = 0
        self.z_hidden_error = 0
        self.z_hidden_difference = 0

    def train_network(self, dataset, output, times):
        for i in range(times):
            self.propagate_forward(dataset)
            self.propagate_backward(dataset, output)

    def propagate_forward(self, dataset):
        self.z_input = numpy.dot(dataset, self.hidden_weight)
        self.z_hidden = activation_function(self.z_input)
        self.z_output = numpy.dot(self.z_hidden, self.output_weight)
        self.forward_propagated = activation_function(self.z_output)

    def propagate_backward(self, dataset, output):
        self.propagation_error = output - self.forward_propagated
        self.propagation_difference = self.propagation_error * gradient_function(self.forward_propagated)
        self.z_hidden_error = self.propagation_difference.dot(self.output_weight.T)
        self.z_hidden_difference = self.z_hidden_error * gradient_function(self.z_hidden)
        self.hidden_weight += dataset.T.dot(self.z_hidden_difference)
        self.output_weight += self.z_hidden.T.dot(self.propagation_difference)
