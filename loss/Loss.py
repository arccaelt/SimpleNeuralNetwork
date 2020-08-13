import abc

class LossFunction(abc.ABC):
    @abc.abstractmethod
    def computeError(self, networkOutput, expectedOutput):
        pass

    @abc.abstractmethod
    def computeErrorUsingDerivative(self, networkOutput, expectedOutput):
        pass