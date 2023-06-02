import numpy as np

class Operations:
    operations_list = [
        "matrix_multiplication",
        "elementwise_addition",
        "elementwise_subtraction",
        "elementwise_multiplication",
        "elementwise_division",
        "transpose",
        "dot_product",
        "sum",
        "mean",
        "max",
        "min",
        "abs",
        "sin",
        "cos",
        "tan",
        "relu",
        "sigmoid",
        "softmax",
        "log",
        "exp",
        "sqrt",
        "power",
        "clip",
        "concatenate",
        "reshape",
        "flatten",
        "slice",
        "pad",
        "convolve",
        "pool",
        # Add more operations as needed
    ]

    @staticmethod
    def matrix_multiplication(inputs):
        return np.matmul(*inputs)

    @staticmethod
    def elementwise_addition(inputs):
        return np.add(*inputs)

    @staticmethod
    def elementwise_subtraction(inputs):
        return np.subtract(*inputs)

    @staticmethod
    def elementwise_multiplication(inputs):
        return np.multiply(*inputs)

    @staticmethod
    def elementwise_division(inputs):
        return np.divide(*inputs)

    @staticmethod
    def transpose(inputs):
        return np.transpose(inputs[0])

    @staticmethod
    def dot_product(inputs):
        return np.dot(*inputs)

    @staticmethod
    def sum(inputs):
        return np.sum(inputs[0])

    @staticmethod
    def mean(inputs):
        return np.mean(inputs[0])

    @staticmethod
    def max(inputs):
        return np.max(inputs[0])

    @staticmethod
    def min(inputs):
        return np.min(inputs[0])

    @staticmethod
    def abs(inputs):
        return np.abs(inputs[0])

    @staticmethod
    def sin(inputs):
        return np.sin(inputs[0])

    @staticmethod
    def cos(inputs):
        return np.cos(inputs[0])

    @staticmethod
    def tan(inputs):
        return np.tan(inputs[0])

    @staticmethod
    def relu(inputs):
        return np.maximum(inputs[0], 0)

    @staticmethod
    def sigmoid(inputs):
        return 1 / (1 + np.exp(-inputs[0]))

    @staticmethod
    def softmax(inputs):
        exp_vals = np.exp(inputs[0])
        return exp_vals / np.sum(exp_vals)

    @staticmethod
    def log(inputs):
        return np.log(inputs[0])

    @staticmethod
    def exp(inputs):
        return np.exp(inputs[0])

    @staticmethod
    def sqrt(inputs):
        return np.sqrt(inputs[0])

    @staticmethod
    def power(inputs):
        return np.power(*inputs)

    @staticmethod
    def clip(inputs):
        return np.clip(*inputs)

    @staticmethod
    def concatenate(inputs):
        return np.concatenate(inputs, axis=0)

    @staticmethod
    def reshape(inputs):
        return np.reshape(inputs[0], inputs[1])

    @staticmethod
    def flatten(inputs):
        return np.reshape(inputs[0], (inputs[0].shape[0], -1))

    @staticmethod
    def slice(inputs):
        return inputs[0][inputs[1]:inputs[2]]

    @staticmethod
    def pad(inputs):
        return np.pad(inputs[0], pad_width=inputs[1], mode='constant')

    @staticmethod
    def convolve(inputs):
        return np.convolve(inputs[0], inputs[1], mode='same')

    @staticmethod
    def pool(inputs):
        return np.max(inputs[0], axis=inputs[1])
