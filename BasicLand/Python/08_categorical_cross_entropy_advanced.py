import numpy as np

# As said in the read me this code is very heavily nnfs book w/ twkz to learn n.n.f.s
netInputs = np.array([
  [0.88, 0.24, 0.14], [0.1, 0.76, 0.41], [0.02, 0.84, 0.08]])
targetOutputs = np.array([
  [1, 0, 0], [0, 1, 0], [0, 1, 0]])

# netInputs = np.array([[0.88, 0.24, 0.14]])
# targetOutputs = np.array([[1, 0, 0]])


class Layer_Dense:
    def __init__(self, _numOfInputs, _numOfNeurons):
      self.weights = 0.10 * np.random.randn(_numOfInputs, _numOfNeurons)
      self.biases = np.zeros((1, _numOfNeurons))
    def forward(self, _inputs):
      self.output = np.dot(_inputs, self.weights) + self.biases

class Activation_ReLU:
  def forward(self, _inputs):
    self.output = np.maximum(0, _inputs)

class Activation_Softmax:
  def forward(self, _inputs):
    exponential_values = np.exp(_inputs - np.max(_inputs, axis=1, keepdims=True))
    normalized_values = exponential_values / np.sum(exponential_values, axis=1, keepdims=True)
    self.output = normalized_values


class Loss:
  def calculate(self, output, y):
    sampleLosses = self.forward(output, y)
    dataLoss = np.mean(sampleLosses)

    return dataLoss

class CategoricalCrossEntropyLoss(Loss):
  def forward(self, yPrediction, yTrue):
    numOfSamples = len(yPrediction)
    """ NNFS note on line/code below
      Clip data to prevent division by 0. 
      Clip both sides to not drag mean towards any value """
    yPredictionClipped = np.clip(yPrediction, 1e-7, 1 - 1e-7)

    if len(yTrue.shape) == 1:
      correctConfidences = yPredictionClipped[
        range(numOfSamples),
        True
      ]
    elif len(yTrue.shape) == 2:
      correctConfidences = np.sum(
        yPredictionClipped*yTrue,
        axis=1
      )
    
    negativeLogLikelihoods = -np.log(correctConfidences)
    return negativeLogLikelihoods


layer1 = Layer_Dense(3,3)
activation1 = Activation_ReLU()

layer2 = Layer_Dense(3,3)
activation2 = Activation_Softmax()

lossFunction = CategoricalCrossEntropyLoss()
layer1.forward(netInputs)
activation1.forward(layer1.output)

layer2.forward(activation1.output)
activation2.forward(layer2.output)
print(activation2.output[:5])

lossVal = lossFunction.calculate(activation2.output, targetOutputs)
print("\nLoss:  ", lossVal)
