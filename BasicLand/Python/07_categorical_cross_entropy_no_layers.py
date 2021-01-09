import numpy as np

netOutput = [0.88, 0.24, 0.14]
targetOutput = [1, 0, 0]

class Loss:
  def calculate(self, output, y):
    sampleLosses = self.forward(output, y)
    dataLoss = np.mean(sampleLosses)

    return dataLoss
  
