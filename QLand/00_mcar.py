import gym
import numpy as np

''' Following "Sentdex" example/teaching w/ own tweaks/additions, etc. '''
env = gym.make("MountainCar-v0")
env.reset()

print('\n env.action_space: ', env.action_space)
print(' env.observation_space: ', env.observation_space)
print(' env.observation_space.high: ', env.observation_space.high)
print(' env.observation_space.low: ', env.observation_space.low, '\n')

discreteObservationSize = [20] * len(env.observation_space.high) # Should not be hardcoded
discreteObservationWindowSize = (
  env.observation_space.high - env.observation_space.low) / discreteObservationSize

print('This is not dynamic as it should be, just MVP')
print(' discreteObservationSize = ([20]*len(env.observation_space.high))\n',  discreteObservationSize, '\n')
print(' discreteObservationWindowSize\n', discreteObservationWindowSize)

''' 
  Init() the qTable as a [20]x[20]x[3]
  This is the size(discreteObservationSize + [env.action_space.n]) 
  (discreteObservationSize & every action possible[3] so (20, 20, 3) -> EveryObservationCombo)
'''
qTable = np.random.uniform(
  low=-2,
  high=0,
  size=(discreteObservationSize + [env.action_space.n])
)
print('\n qTable.shape\n', qTable.shape)
# print('qTable\n', qTable, '\n')

'''
isDone = False
while not isDone:
  action = 2
  currState, myReward, isDone, _ = env.step(action)
  print(myReward, currState)
  env.render()

env.close()
'''