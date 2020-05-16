from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPool2D, Activation, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import TensorBoard

from collections import deque # Need to understand this pkg better
'''
  Using tf-2.2 rather then ~tf-1.0 like tut
  Following Tutorial: https://pythonprogramming.net/deep-q-learning-dqn-reinforcement-learning-python-tutorial/
'''

class DqlAgent:
  def __init__(self):
    self.model = self.create_model() # Gets trained every step
    
    self.targetModel = self.create_model() # Gets predicted against every step
    self.targetModel.set_weights(self.model.get_weights())


  def create_model(self):
    model = Sequential()
    model.add(Conv2D(256, (3,3), input_shape=envObservationSpaceVals))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(2,2))
    model.add(Dropout(0.2))

    model.add(Conv2D(256, (3,3)))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(2,2))
    model.add(Dropout(0.2))

    model.add(Flatten())
    model.add(Dense(64))

    model.add(Dense(envActionSpaceSize, activation="linear"))
    model.compile(Loss="mse", optimizer=Adam(Lr=0.001), metrics=['accuracy'])
    return model

