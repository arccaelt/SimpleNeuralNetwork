import numpy as np

class Neuron:
    def __init__(self, weights: np.ndarray):
        self._validateWeights(weights)
        self.weights = weights

    def _validateWeights(self, weights):
        weightsShape = weights.shape

        if len(weightsShape) != 2 or weightsShape[0] > 1:
            raise ValueError("The neuron's weights must be in the form (1, weights_count)")

    def getWeight(self, index):
        return self.weights[0, index]

    def getWeights(self):
        return self.weights

    def computeWeightedSum(self, previousValues):
        return np.sum(np.dot(self.weights, previousValues))

    def getWeigthsCount(self):
        return len(self.weights)

    def updateWeight(self, weightIndex, newValue):
        self.weights[weightIndex] = newValue