import numpy as np
from activation.base_activation import Activation

class SigmoidActivation(Activation):
    def getOutputValue(self, neuronWeightedSum):
        return 1 / (1 + np.exp(-neuronWeightedSum))

    def getDerivativeOutputValue(self, neuronWeightedSum):
        regularOutput = self.getOutputValue(neuronWeightedSum)
        return regularOutput * (1 - regularOutput)