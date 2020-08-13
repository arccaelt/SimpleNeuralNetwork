import network
from utils import printDebuggingOutput
from layers import denselayer
from activation.SigmoidActivation import SigmoidActivation
from loss.MSE import MSE

train_data = [[1, 0], [0, 1], [1, 1], [0, 0]]
output_data = [1, 1, 0, 0]

model = network.Network()
model.addDenseLayer(denselayer.DenseLayer(2, 2, SigmoidActivation()))
model.addDenseLayer(denselayer.DenseLayer(1, 2, SigmoidActivation()))
printDebuggingOutput(model)
model.train(500, 0.1, train_data, output_data, MSE())
# result = model.feedforward([1, 2])
# print("Result: {}".format(result))

printDebuggingOutput(model)