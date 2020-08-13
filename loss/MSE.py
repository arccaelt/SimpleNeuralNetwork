import numpy as np
from loss import Loss

class MSE(Loss.LossFunction):
    def computeError(self, networkOutput, expectedOutput):
        n = len(networkOutput)
        valueDifference = networkOutput - expectedOutput
        squaredDiffeence = valueDifference ** 2
        return (1 / n) * np.sum(squaredDiffeence)

    def computeErrorUsingDerivative(self, variableUsedForDerivation, expectedOutput):
        return expectedOutput - variableUsedForDerivation
