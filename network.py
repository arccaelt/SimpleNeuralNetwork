import numpy as np
from layers.denselayer import DenseLayer

class Network:
    def __init__(self):
        self.layers = []

    def addDenseLayer(self, layer: DenseLayer):
        self.layers.append(layer)

    def feedforward(self, inputData):
        previousData = inputData
        for layer in self.layers:
            previousData = layer.computeLayersOutput(previousData)
        return previousData[0][0]

    def train(self, epochsLimit, learning_rate, trainingData, desiredValues, lossFunction):
        self.lossFunction = lossFunction

        for currentEpoch in range(epochsLimit):
            for idx, input_data in enumerate(trainingData):
                networkOutput = self._feedforwardForTraining(input_data)
                self._computeDeltasForOutputLayer(desiredValues[idx])
                self._computeHiddenLayerDeltaVector()
                self._updateWeights(learning_rate)

            self._printTrainingInformation(currentEpoch, epochsLimit)

    def _printTrainingInformation(self, currentEpoch, totalEpochsCount):
        print("training epoch: {}/{}".format(currentEpoch, totalEpochsCount))

    def _feedforwardForTraining(self, inputData):
        previousData = inputData
        for idx, layer in enumerate(self.layers):
            layer.computeLayerBackpropagationInformation(previousData)
            previousData = layer.activationValuesVector

        return previousData

    def _computeDeltasForOutputLayer(self, desiredValue):
        outputLayer = self._getOutputLayer()
        partialDerivativesVector = self._computeOutputLayerLossFunctionPartialDerivatives(outputLayer, desiredValue)
        outputLayer.deltasVector = partialDerivativesVector * outputLayer.derivativeOfActivationFunctionValues

    def _computeOutputLayerLossFunctionPartialDerivatives(self, outputLayer, desiredValue):
        lossFunctionPartialDerivatives = self.lossFunction.computeErrorUsingDerivative(outputLayer.activationValuesVector, np.array(desiredValue))
        return lossFunctionPartialDerivatives

    def _getOutputLayer(self):
        layersCount = len(self.layers)
        return self.layers[layersCount - 1]

    def hasHiddenLayers(self):
        return len(self.layers) > 1

    def _updateWeights(self, learning_rate):
        if not self.hasHiddenLayers():
            return

        layersCount = len(self.layers)

        for layerIndex in range(layersCount - 1, -1, -1):
            currentLayer = self.layers[layerIndex]
            for neuronIndex in range(currentLayer.getNeuronsCount()):
                for weightIndex in range(currentLayer.getPreviousLayerNeuronCount()):
                    weightValue = currentLayer.getWeight(neuronIndex, weightIndex)
                    gradient = weightIndex * currentLayer.deltasVector[neuronIndex]
                    updatedValue = weightValue - learning_rate * gradient[0]
                    currentLayer.updateWeight(neuronIndex, weightIndex, updatedValue)

    def isOutputLayer(self, layerIndex):
        return layerIndex + 1 == len(self.layers)

    def _computeHiddenLayerDeltaVector(self):
        if not self.hasHiddenLayers():
            return

        layersCount = len(self.layers)
        for idx in range(layersCount - 2, -1, -1):
            currentLayer = self.layers[idx]
            nextLayer = self.layers[idx + 1]
            weightsAndDeltas = nextLayer.weightsMatrix.T * nextLayer.deltasVector
            currentLayer.deltasVector = weightsAndDeltas * currentLayer.derivativeOfActivationFunctionValues