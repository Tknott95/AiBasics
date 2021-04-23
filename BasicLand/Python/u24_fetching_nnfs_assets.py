import os
import urllib
import urllib.request

from zipfile import ZipFile

"""
  - SCRIPT PULLS NNFS .zip DATA. Using book conventions.
  Libs for pulling my zip and decompressing it
"""
DATA_URL = 'https://nnfs.io/datasets/fashion_mnist_images.zip'
DATA_FILE = 'fashion_mnist_images.zip'
DATA_FOLDER = 'Assets/fashion_mnist_images'


def fetchData():
  if not os.path.isfile(DATA_FILE):
    print(f'\n  Downloading {DATA_URL} and saving as {DATA_FILE}...')
    urllib.request.urlretrieve(DATA_URL, DATA_FILE)
  
  print('\n  Decompressing Images...')
  with ZipFile(DATA_FILE) as zipImages:
    zipImages.extractall(DATA_FOLDER)


fetchData()
