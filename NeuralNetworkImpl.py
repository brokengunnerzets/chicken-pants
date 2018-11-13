import numpy;


def activation_function(x):
    # Sigmoid
    return 1 / (1 + numpy.exp(-x))


def gradient_function(x):
    # Sigmoid derivative
    return x * (1 - x)


def get_random_weight(x, y):
    return numpy.random.rand(x, y)


class NeuralNetworkImpl:
    def __init__(self, input_size, hidden_size, output_size):
        self.param_input_size = input_size
        self.param_hidden_size = hidden_size
        self.param_output_size = output_size
        self.hidden_weight = 0
        self.output_weight = 0
        self.forward_propagated = 0
        self.z_hidden = 0
        self.z_output = 0
        self.propagation_difference = 0
        self.z_hidden_difference = 0

    def define_weights(self):
        self.hidden_weight = get_random_weight(self.param_input_size, self.param_hidden_size)
        self.output_weight = get_random_weight(self.param_hidden_size, self.param_output_size)

    def update_zs(self, input_data):
        self.z_hidden = activation_function(numpy.dot(input_data, self.hidden_weight))
        self.z_output = numpy.dot(self.z_hidden, self.output_weight)

    def update_weights(self, input_data):
        self.hidden_weight = self.hidden_weight + input_data.T.dot(self.z_hidden_difference)
        self.output_weight = self.output_weight + self.z_hidden.T.dot(self.propagation_difference)

    def update_difference(self, output_data):
        self.propagation_difference = (output_data - self.forward_propagated) * gradient_function(self.forward_propagated)
        self.z_hidden_difference = (self.propagation_difference.dot(self.output_weight.T)) * gradient_function(self.z_hidden)

    def propagate_forward(self, input_data):
        self.update_zs(input_data)
        self.forward_propagated = activation_function(self.z_output)

    def propagate_backward(self, input_data, output):
        self.update_difference(output)
        self.update_weights(input_data)

    def train_network(self, input_data, output, times):
        for i in range(times):
            self.propagate_forward(input_data)
            self.propagate_backward(input_data, output)

    def predict(self, predicting):
        self.propagate_forward(predicting)
        return self.forward_propagated
