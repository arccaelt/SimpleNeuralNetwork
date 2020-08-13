import abc

class Activation(abc.ABC):

    @abc.abstractmethod
    def getOutputValue(self, neuronWeightedSum):
        pass

    @abc.abstractmethod
    def getDerivativeOutputValue(self, neuronWeightedSum):
        pass