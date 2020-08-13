from activation import base_activation

class ReLU(base_activation.Activation):
    def getOutputValue(self, neuronWeightedSum):
        return max(neuronWeightedSum, 0)

    def getDerivativeOutputValue(self, neuronWeightedSum):
        return 1 if neuronWeightedSum >= 0 else 0

