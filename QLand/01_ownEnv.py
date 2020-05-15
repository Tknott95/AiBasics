import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
from matplotlib import style
import pickle
import time

''' 
  Following "Sentdex" example/teaching w/ own tweaks/additions, etc.
  TutLink: https://pythonprogramming.net/own-environment-q-learning-reinforcement-learning-python-tutorial/   
'''
def main():
  style.use("ggplot")

  gridSize = 10
  agentEpisodes = 24000
  movePenalty = 1
  enemyPenalty = 300
  foodReward = 25

  epsilon = 0 # 0.9 - if no qtable data
  epsilonDecay = 0.9998
  showEveryEps = 2000  #  1  # 2000 - if no qtable data

  startQTable = "qTable-1589559420.pickle" # None - if no qtable data

  learningRate = 0.1
  discount = 0.95

  playerN = 1
  foodN = 2
  enemyN = 3

  dictionary = {1: (255, 175, 0), 2: (0, 255, 0), 3: (0, 0, 255)} # BGR format


  class Blob:
    def __init__(self):
      self.x = np.random.randint(0, gridSize)
      self.y = np.random.randint(0, gridSize)
    def __str__(self):
      return f"{self.x}, {self.y}"
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
      elif self.x > gridSize-1:
        self.x = gridSize-1
      if self.y < 0:
        self.y = 0
      elif self.y > gridSize-1:
        self.y = gridSize-1

  if startQTable is None: # Creating qTable if ! have one
    qTable = {}
    # Every combo
    for x1 in range(-gridSize+1, gridSize):
      for y1 in range(-gridSize+1, gridSize):
        for x2 in range(-gridSize+1, gridSize):
          for y2 in range(-gridSize+1, gridSize):
            qTable[((x1, y1), (x2, y2))] = [np.random.uniform(-5,0) for i in range(4)]
  else:
    with open(startQTable, "rb") as f: # Loading pretrained file if have one.
      qTable = pickle.load(f)

  episodeRewards = []
  for episode in range(agentEpisodes):
    player = Blob()
    food = Blob()
    enemy = Blob()

    if episode % showEveryEps == 0:
      print(f"on # {episode}, epsilon: {epsilon}")
      print(f"{showEveryEps} ep mean {np.mean(episodeRewards[-showEveryEps:])}")
      show = True
    else:
      show = False
    
    episodeReward = 0
    for i in range(200): #stepsToTake is 200
      observation = (player-food, player-enemy)
      if np.random.random() > epsilon:
        action = np.argmax(qTable[observation])
      else:
        action = np.random.randint(0, 4)
      
      player.action(action)

      if player.x == enemy.x and player.y == enemy.y:
        reward = -enemyPenalty
      elif player.x == food.x and player.y == food.y:
        reward = foodReward
      else:
        reward = -movePenalty
      
      newObservation = (player-food, player-enemy)
      maxFutureQ = np.max(qTable[newObservation])
      currentQ = qTable[observation][action]

      if reward == foodReward:
        newQ = foodReward
      elif reward == -enemyPenalty:
        newQ = -enemyPenalty
      else:
        newQ = (1-learningRate) * currentQ + learningRate * (reward + discount * maxFutureQ) # Actual qLearning Algo
      
      qTable[observation][action] = newQ

      if show:
        # Making env, for custom basic aF "blob" game
        env = np.zeros((gridSize, gridSize, 3), dtype=np.uint8)
        env[food.y][food.x] = dictionary[foodN] # Flipped x,y to y,x
        env[player.y][player.x] = dictionary[playerN] # Flipped x,y to y,x
        env[enemy.y][enemy.x] = dictionary[enemyN] # Flipped x,y to y,x

        img = Image.fromarray(env, "RGB")
        img = img.resize((350, 350))
        cv2.imshow("Q BlobZ", np.array(img))

        if reward == foodReward or reward == -enemyPenalty: # if messed up or hit the enemy
          if cv2.waitKey(500) & 0xFF == ord("q"): # If q key it breaks
            break
        else:
          if cv2.waitKey(1) & 0xFF == ord("q"):
            break
      
      episodeReward += reward
      if reward == foodReward or reward == -enemyPenalty:
        break
    
    episodeRewards.append(episodeReward)
    epsilon *= epsilonDecay
  
  movingAverage = np.convolve(episodeRewards, np.ones((showEveryEps,)) / showEveryEps, mode="valid")

  plt.plot([i for i in range(len(movingAverage))], movingAverage)
  plt.ylabel(f"award {showEveryEps}ma")
  plt.xlabel("eps #")
  plt.show()

  with open(f"qTable-{int(time.time())}.pickle", "wb") as f:
    pickle.dump(qTable, f)


if __name__ == "__main__":
  main()


