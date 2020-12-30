from unittest import TestCase

from network import Network
import matplotlib.pyplot as plt

class Testnetwork(TestCase):
    def setUp(self):
        self.layers = [2, 4, 4, 2]
        self.net = Network(self.layers)

    def test_initialize_weights(self):
        self.assertEqual(len(self.net.weights), self.net.L - 1)
        for weight, layer_in, layer_out in zip(self.net.weights, self.net.layers[:-1], self.net.layers[1:]):
            self.assertEqual(weight.shape, (layer_in, layer_out))

    def test_initialize_biases(self):
        self.assertEqual(len(self.net.biases), self.net.L - 1)
        for b in self.net.biases:
            self.assertEqual(len(b), 1)

