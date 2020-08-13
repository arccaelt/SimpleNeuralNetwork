import numpy as np

def getEmptyColumnVector(rowsCount):
    return np.empty((rowsCount, 1))

def printDebuggingOutput(model):
    for idx, layer in enumerate(model.layers):
        print("Layer {}".format(idx))

        for neuronIndex, neuron in enumerate(layer.neurons):
            print("\tNeuron {}".format(neuronIndex))
            print("\t\t{}".format(neuron.getWeights()))