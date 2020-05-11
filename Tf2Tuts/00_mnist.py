import tensorflow as tf
from tensorflow import keras
import numpy as np
# import matplotlib.pylab as plt <- larger lib, pyplot is standard way
from matplotlib import pyplot as plt

'''
  Switch w/ matplotlib.plypot on linux
  matplotlib w/out jupyter use -> plt.show()
'''

def main():
  print("TF-Verison: ", tf.__version__)

  mnist_dataset = keras.datasets.fashion_mnist
  (train_images, train_labels), (test_images, test_labels) = mnist_dataset.load_data()
  '''
    The images are 28x28 NumPy arrays, with pixel values ranging from 0 to 255.
    The labels are an array of integers, ranging from 0 to 9.
    These correspond to the class of clothing the image represents:
  '''
  class_names = [
    'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
    'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
  ]

  """ EXPLORE THE DATA """
  print("\ntrain_images.shape: ", train_images.shape)     # (60000, 28, 28)
  print("len(train_labels)): ", len(train_labels))        # 60000
  print("train_labels: ", train_labels)                   # [9 0 0 ... 3 0 5] - each an int between 0-9
  print('test_images.shape: ', test_images.shape)         # (10000, 28, 28)
  print('len(test_labels): ', len(test_labels))           # 10000

  """ PREPROCESS THE DATA """
  plt.figure()
  plt.imshow(train_images[0])
  plt.colorbar()
  plt.grid(False)
  plt.show()


def testPlt():
  x = np.arange(0, 5, 0.1)
  y = np.sin(x)
  plt.plot(x, y)

  plt.tight_layout()
  plt.show()


if __name__ == '__main__':
  main()
