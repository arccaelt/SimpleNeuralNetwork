import unittest
import numpy as np
from layers import denselayer
from activation.SigmoidActivation import SigmoidActivation


class DenseLayerTests(unittest.TestCase):
    TEST_LAYER_NEURONS_COUNT = 2
    TEST_LAYER_PREVIOUS_LAYER_NEURONS_COUNT = 2
    TEST_LAYER = denselayer.DenseLayer(TEST_LAYER_NEURONS_COUNT,
                                       TEST_LAYER_PREVIOUS_LAYER_NEURONS_COUNT,
                                       SigmoidActivation())

    TEST_LAYER_WEIGHTED_SUM_FIELD_NAME = "weightedSumVector"
    TEST_LAYER_ACTIVATION_FIELD_NAME = "activationValuesVector"
    TEST_LAYER_ACTIVATION_DERIVATIVE_FIELD_NAME = "derivativeOfActivationFunctionValues"

    def testLayerReturnsNeuronsCountCorrectly(self):
        self.assertEqual(self.TEST_LAYER.getNeuronsCount(), self.TEST_LAYER_NEURONS_COUNT)

    def testLayerReturnsPreviousLayerNeuronsCountCorrectly(self):
        self.assertEqual(self.TEST_LAYER.getPreviousLayerNeuronCount(), self.TEST_LAYER_PREVIOUS_LAYER_NEURONS_COUNT)

    def testLayerComputesNeuronOutputCorrectly(self):
        neuronTestValue = 15
        sigmoidActivation = SigmoidActivation()
        actualNeuronValue = sigmoidActivation.getOutputValue(neuronTestValue)
        self.assertEqual(self.TEST_LAYER.getNeuronOutput(neuronTestValue), actualNeuronValue)

    def testWeightsMatrixHasTheRightSize(self):
        layerWeightsMatrix = self.TEST_LAYER.weightsMatrix.shape
        self.assertEqual(layerWeightsMatrix[0], self.TEST_LAYER_NEURONS_COUNT)
        self.assertEqual(layerWeightsMatrix[1], self.TEST_LAYER_PREVIOUS_LAYER_NEURONS_COUNT)

    def testWeightRetrivalIsCorrect(self):
        retrievedWeight = self.TEST_LAYER.getWeight(0, 0)
        actualWeight = self.TEST_LAYER.weightsMatrix[0, 0]
        self.assertEqual(retrievedWeight, actualWeight)

    def testWeightUpdateIsCorrect(self):
        self.TEST_LAYER.updateWeight(0, 0, 0)
        self.assertEqual(self.TEST_LAYER.getWeight(0, 0), 0)

    def testComputingLayersOutputHasTheRightNumberOfNeurons(self):
        layerOutput = self.TEST_LAYER.computeLayersOutput(self._getRandomPreviousLayerInformation())
        self.assertEqual(len(layerOutput), self.TEST_LAYER_NEURONS_COUNT)

    def _getRandomPreviousLayerInformation(self):
        randomPreviousLayerActivationValues = np.random.random((self.TEST_LAYER_PREVIOUS_LAYER_NEURONS_COUNT, 3))
        return randomPreviousLayerActivationValues

    def testComputingBackpropagationInformationCreatesFields(self):
        self.TEST_LAYER.computeLayerBackpropagationInformation(self._getRandomPreviousLayerInformation())
        self.assertTrue(hasattr(self.TEST_LAYER, self.TEST_LAYER_WEIGHTED_SUM_FIELD_NAME))
        self.assertTrue(hasattr(self.TEST_LAYER, self.TEST_LAYER_ACTIVATION_FIELD_NAME))
        self.assertTrue(hasattr(self.TEST_LAYER, self.TEST_LAYER_ACTIVATION_DERIVATIVE_FIELD_NAME))