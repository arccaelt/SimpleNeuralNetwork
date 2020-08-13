import neuron
import unittest
import numpy as np

class NeuronTests(unittest.TestCase):
    NEURON_TEST_DATA = [3, 4]
    neuron = neuron.Neuron(np.random.random((1, 2)))

    def testNeuronOutput(self):
        neuronOutput = self.neuron.computeWeightedSum(np.array(self.NEURON_TEST_DATA))
        self.assertEqual(neuronOutput, self._computeNeuronWeightedSum())

    def _computeNeuronWeightedSum(self):
        weightedSum = self.NEURON_TEST_DATA[0] * self.neuron.getWeight(0) + self.NEURON_TEST_DATA[1] * self.neuron.getWeight(1)
        return weightedSum

    def testNeuronWeightsValidation(self):
        self.assertRaises(ValueError, neuron.Neuron, np.random.random((5, 2)))
        self.assertRaises(ValueError, neuron.Neuron, np.random.random((1, 1, 1)))

