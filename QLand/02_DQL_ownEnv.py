from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPool2D, Activation, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import TensorBoard
import numpy as np

from collections import deque # Need to understand this pkg better - list w/ max size set .. etc
import time
import random
'''
  Not a fan of this abstraction pattern, tf2 docs have better ways, may be due to tf1? Will create my own in a diff way 
  Will do one w/ my own env + tf2.2 conventions

  Using tf-2.2 rather then ~tf-1.0 like tut
  Following Tutorial: https://pythonprogramming.net/deep-q-learning-dqn-reinforcement-learning-python-tutorial/
'''

REPLAY_MEM_SIZE = 50_000 # Normally hate caps yet will set like a c++ const as the tut does. Underscore works like a "comma"
MIN_REPLAY_MEM_SIZE = 1_000
MODEL_NAME = '256x2' # it is a 256x2 conv net
MINIBATCH_SIZE = 64
DISCOUNT = 0.99
UPDATE_TARGET_EVERY = 5

class Blob:
  def __init__(self, size):
    self.x = np.random.randint(0, gridSize)
    self.y = np.random.randint(0, gridSize)
  def __str__(self):
    return f"Blob ({self.x}, {self.y})"
  def __sub__(self, other):
    return (self.x - other.x, self.y - other.y)
    
  def action(self, choice):
    if choice == 0:
      self.move(x=1, y=1)
    elif choice == 1:
      self.move(x=-1, y=-1)
    elif choice == 2:
      self.move(x=-1, y=1)
    elif choice == 3:
      self.move(x=1, y=-1)
    elif choice == 4:
      self.move(x=1, y=0)
    elif choice == 5:
      self.move(x=-1, y=0)
    elif choice == 6:
      self.move(x=0, y=1)
    elif choice == 7:
      self.move(x=0, y=-1)
    elif choice == 8:
      self.move(x=0, y=0)

  def move(self, x=False, y=False):
    if not x:
      self.x += np.random.randint(-1, 2)
    else:
      self.x += x
    if not y:
      self.y += np.random.randint(-1, 2)
    else:
      self.y += y

    if self.x < 0: # Boundaries, clearly
      self.x = 0
    elif self.x > self.size-1:
      self.x = self.size-1
    if self.y < 0:
      self.y = 0
    elif self.y > self.size-1:
      self.y = self.size-1

class BlobEnv:
    SIZE = 10
    RETURN_IMAGES = True
    MOVE_PENALTY = 1
    ENEMY_PENALTY = 300
    FOOD_REWARD = 25
    OBSERVATION_SPACE_VALUES = (SIZE, SIZE, 3)  # 4
    ACTION_SPACE_SIZE = 9
    PLAYER_N = 1  # player key in dict
    FOOD_N = 2  # food key in dict
    ENEMY_N = 3  # enemy key in dict
    # the dict! (colors)
    d = {1: (255, 175, 0),
         2: (0, 255, 0),
         3: (0, 0, 255)}

    def reset(self):
        self.player = Blob(self.SIZE)
        self.food = Blob(self.SIZE)
        while self.food == self.player:
            self.food = Blob(self.SIZE)
        self.enemy = Blob(self.SIZE)
        while self.enemy == self.player or self.enemy == self.food:
            self.enemy = Blob(self.SIZE)

        self.episode_step = 0

        if self.RETURN_IMAGES:
            observation = np.array(self.get_image())
        else:
            observation = (self.player-self.food) + (self.player-self.enemy)
        return observation

    def step(self, action):
        self.episode_step += 1
        self.player.action(action)

        #### MAYBE ###
        #enemy.move()
        #food.move()
        ##############

        if self.RETURN_IMAGES:
            new_observation = np.array(self.get_image())
        else:
            new_observation = (self.player-self.food) + (self.player-self.enemy)

        if self.player == self.enemy:
            reward = -self.ENEMY_PENALTY
        elif self.player == self.food:
            reward = self.FOOD_REWARD
        else:
            reward = -self.MOVE_PENALTY

        done = False
        if reward == self.FOOD_REWARD or reward == -self.ENEMY_PENALTY or self.episode_step >= 200:
            done = True

        return new_observation, reward, done

    def render(self):
        img = self.get_image()
        img = img.resize((300, 300))  # resizing so we can see our agent in all its glory.
        cv2.imshow("image", np.array(img))  # show it!
        cv2.waitKey(1)

    # FOR CNN #
    def get_image(self):
        env = np.zeros((self.SIZE, self.SIZE, 3), dtype=np.uint8)  # starts an rbg of our size
        env[self.food.x][self.food.y] = self.d[self.FOOD_N]  # sets the food location tile to green color
        env[self.enemy.x][self.enemy.y] = self.d[self.ENEMY_N]  # sets the enemy location to red
        env[self.player.x][self.player.y] = self.d[self.PLAYER_N]  # sets the player tile to blue
        img = Image.fromarray(env, 'RGB')  # reading to rgb. Apparently. Even tho color definitions are bgr. ???
        return img


env = BlobEnv()

# For stats
ep_rewards = [-200]

# For more repetitive results
random.seed(1)
np.random.seed(1)
tf.set_random_seed(1)

# Memory fraction, used mostly when trai8ning multiple agents
#gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=MEMORY_FRACTION)
#backend.set_session(tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)))

# Create models folder
if not os.path.isdir('models'):
    os.makedirs('models')


# keras wants to update log file every .fit so this will allow keras not to create a new log every .fit
# tf2.2 might not need this, looks like shit code
class ModifiedTensorBoard(TensorBoard):
  def __init__(self, **kwargs):
    super().__init__(**kwargs)
    self.step = 1
    self.writer = tf.summary.FileWriter(self.log_dir)
  def set_model(self, model):
    pass
  def on_epoch_end(self, epoch, logs=None):
    self.update_stats(**logs)
  def on_batch_end(self, batch, logs=None):
    pass
  def on_train_end(self, _):
    pass
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

  def updateReplayMemory(self, transition):
    self.replayMemory.append(transition) # Allows for newQ form. In short transition is:  ObsSpace, action, reward.. new ObsSpace, isDone?

  def getQVals(self, terminalState, step):
    return self.modelPredict(np.array(state).reshape(-1, *state.shape)/255)[0] # /255 "scale" the rgb data getting passed in, as env is setup as such

  def train(self, terminalState, step):
    if len(self.replayMemory) < MIN_REPLAY_MEM_SIZE:
      return
    
    minibatch = random.sample(self.replayMemory, MINIBATCH_SIZE)
    currStates = np.array(transition[0] for transition in minibatch])/255 # /255 "scale" the rgb data getting passed in, as env is setup as such
    currQValsList = self.model.predict(currStates)

    newCurrStates = np.array([transition[3] for transition in minibatch])/255 # /255 "scale" the rgb data getting passed in, as env is setup as such
    futureQValsList = self.targetModel.predict(newCurrStates)

    X = []
    Y = []

    for index, (currState, action, reward, newCurrState, done) in enumerate(minibatch):
      if not done:
        maxFutureQVal = np.max(futureQValsList[index])
        newQVal = reward + DISCOUNT * maxFutureQVal
      else:
        newQVal = reward
    
      currQVals = currQValsList[index]
      currQVals[action] = newQVal

      X.append(currState)
      Y.append(currQVals)
    
    self.model.fit(np.array(X)/255, np.array(Y), batch_size=MINIBATCH_SIZE, verbose=0, shuffle=False, callbacks=[self.tensorboard] if terminalState else None)
    
    if terminalState:
      self.targetUpdateCounter += 1
    if self.targetUpdateCounter > UPDATE_TARGET_EVERY:
      self.targetModel.set_weights(self.model.get_weights())
      self.targetUpdateCounter = 0


