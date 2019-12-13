import numpy as np
import math
from numpy.ma import tanh as tanh_function, exp


def sigmoid(input):
    exp
    return exp(input) / (1 + exp(input))


def relu(input):
    return input * (input > 0)

def linear(input):
    return input

def tanh(input):
    return tanh_function(input)
