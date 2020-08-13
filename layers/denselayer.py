import numpy as np
from neuron import Neuron
from activation.base_activation import Activation
from utils import getEmptyColumnVector

class DenseLayer:
    def __init__(self, neuronsCount: int, previousLayerNeuronsCount: int, activationFunction: Activation):
        self._initializeNeurons(neuronsCount, previousLayerNeuronsCount)
        self.previousLayerNeuronsCount = previousLayerNeuronsCount
        self.activationFunction = activationFunction

    def _initializeNeurons(self, neuronsCount: int, previousDimension: int):
        self.neurons = []
        self.weightsMatrix = np.random.random((neuronsCount, previousDimension))
        for i in range(neuronsCount):
            self.neurons.append(Neuron(self.weightsMatrix[i, :].reshape(1, -1)))

    def getWeightsMatrix(self):
        return self.weightsMatrix

    def updateWeight(self, neuronIndex, weightIndex, newValue):
        self.weightsMatrix[neuronIndex, weightIndex] = newValue

    def getWeight(self, neuronIndex, weightIndex):
        return self.weightsMatrix[neuronIndex, weightIndex]

    def _getNeuronWithRandomValues(self, previousDimension: int):
        weights = np.random.random((1, previousDimension))
        return Neuron(weights)

    def computeLayersOutput(self, previousLayerOutput):
        outputValues = getEmptyColumnVector(self.getNeuronsCount())
        for idx, neuron in enumerate(self.neurons):
            neuronWeightedSum = self.neurons[idx].computeWeightedSum(previousLayerOutput)
            outputValues[idx] = self.activationFunction.getOutputValue(neuronWeightedSum)
        return outputValues

    def computeLayerBackpropagationInformation(self, previousLayerOutput):
        neuronsCount = self.getNeuronsCount()
        self.weightedSumVector = getEmptyColumnVector(neuronsCount)
        self.activationValuesVector = getEmptyColumnVector(neuronsCount)
        self.derivativeOfActivationFunctionValues = getEmptyColumnVector(neuronsCount)

        for idx, neuron in enumerate(self.neurons):
            self.weightedSumVector[idx] = self.neurons[idx].computeWeightedSum(previousLayerOutput)
            self.activationValuesVector[idx] = self.activationFunction.getOutputValue(self.weightedSumVector[idx])
            self.derivativeOfActivationFunctionValues[idx] = self.activationFunction.getDerivativeOutputValue(self.weightedSumVector[idx])

    def getNeuronOutput(self, neuronValue):
        return self.activationFunction.getOutputValue(neuronValue)

    def getPreviousLayerNeuronCount(self):
        return self.previousLayerNeuronsCount

    def getNeuronsCount(self) -> int:
        return len(self.neurons)

