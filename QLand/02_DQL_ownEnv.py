from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPool2D, Activation, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import TensorBoard

from collections import deque # Need to understand this pkg better - list w/ max size set .. etc
import time
'''
  Will do one w/ my own env + tf2.2 conventions
  Using tf-2.2 rather then ~tf-1.0 like tut
  Following Tutorial: https://pythonprogramming.net/deep-q-learning-dqn-reinforcement-learning-python-tutorial/
'''

REPLAY_MEM_SIZE = 50_000 # Normally hate caps yet will set like a c++ const as the tut does. Underscore works like a "comma"
MODEL_NAME = '256x2' # it is a 256x2 conv net



  # Custom method for saving own metrics
  # Creates writer, writes custom metrics and closes writer
  def update_stats(self, **stats):
    self._write_logs(stats, self.step)

class DqlAgent:
  def __init__(self):
    self.model = self.createModel() # Gets trained every step
    
    self.targetModel = self.createModel() # Gets predicted against every step
    self.targetModel.set_weights(self.model.get_weights())

    self.replayMemory = deque(maxlen=REPLAY_MEM_SIZE)
    self.tensorboard = ModifiedTensorBoard(Log_dir=f"logs/{MODEL_NAME}-{int(time.time())}")
    
    self.targetUpdateCounter = 0


  def createModel(self):
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

