import functools
import numpy as np
import tensorflow as tf

'''
  @TITLE
    Would (I|Mock_Person) survive the titanic?
  @DOING
   Will be loading pre-made csv data, from daZ g00gle, and playing w/ it.
  @CREDS
    Will have my own tweaks so think of this as a "fork" in a way.
    From tensorflow.ord docs,original author of base code tfTeam.
    Tweaked code from: url(https://www.tensorflow.org/tutorials/load_data/csv)
''' 

def main():
  # Will use my own lcoal files for nexxt csv example. Following tf2 coding conventions prior
  # Have done this type of coding multiple times yet not with tf2 and new conventions
  TRAIN_DATA_URL = "https://storage.googleapis.com/tf-datasets/titanic/train.csv"
  TEST_DATA_URL = "https://storage.googleapis.com/tf-datasets/titanic/eval.csv"

  train_file_path = tf.keras.utils.get_file("train.csv", TRAIN_DATA_URL)
  test_file_path = tf.keras.utils.get_file("eval.csv", TEST_DATA_URL)

  # This makes np vals easier to read, @TODO refer to docs and understand why as I have never done this
  np.set_printoptions(precision=2, suppress=True)

  # Viewing how CSV is formatted w/ !head {train_file_path} from docs must be a jupyter func
  # ONLY A JUPYTER FUNC: !head {train_file_path} - Will just create my own to reuse
  ''' SHOWS CSV HEAD AS
    survived,sex,age,n_siblings_spouses,parch,fare,class,deck,embark_town,alone
    0,male,22.0,1,0,7.25,Third,unknown,Southampton,n
    1,female,38.0,1,0,71.2833,First,C,Cherbourg,n
    1,female,26.0,0,0,7.925,Third,unknown,Southampton,y
    1,female,35.0,1,0,53.1,First,C,Southampton,n
    0,male,28.0,0,0,8.4583,Third,unknown,Queenstown,y
    0,male,2.0,3,1,21.075,Third,unknown,Southampton,n
    1,female,27.0,0,2,11.1333,Third,unknown,Southampton,n
    1,female,14.0,1,0,30.0708,Second,unknown,Cherbourg,n
    1,female,4.0,1,1,16.7,Third,G,Southampton,n
  '''
  # Only col I will need w/ label is the one I am predicting, which is: WillISurvive?
  # Following tf2 conventions, aka CAPS
  COL_LABEL = 'survived'
  LABELS = [0,1]

  # Using Custom Function: get_dataset(path, **kwargs)
  raw_train_data = get_dataset(COL_LABEL, train_file_path)
  raw_test_data = get_dataset(COL_LABEL, test_file_path)

  show_batch(raw_train_data)
  
  # Set col labels after seeing data each batches label
  '''
    This is only if data has no col names.
    If this is the case then pass in, column_names=MY_COL_LABELS_FROM_CSV
      MY_COL_LABELS_FROM_CSV = ['survived', 'sex', 'age', ...]
      get_dataset(_path, column_names=MY_COL_LABELS_FROM_CSV)
  '''
  temp_dataset = get_dataset(COL_LABEL, train_file_path)
  show_batch(temp_dataset)

  '''  
    If you need to omit some cols from the data:
      create list on only cols you want to use
      pass into the select_columns arg of the constructor.
  '''
  COLS_TO_USE = [
    'survived', 'age', 'n_siblings_spouses',
    'class', 'deck', 'alone']
  temp_dataset = get_dataset(COL_LABEL, train_file_path, select_columns=COLS_TO_USE)
  show_batch(temp_dataset)
  # DATA FINISHED BEING LOADED IN

  # Now to preprocess the data
  '''
    CSV files can contain many diff data types.
    Usually you want to convert these from mixed types to a fixed length vector prior to being fed to the model.
    tf.feature_column is a built in tf func for input conversions.
    can prepoc data w/out tf and jsut pass it in, obviously.
    Doing preproc in model allows for when you export model it includes preproc.
     This allows for passing raw data directly to model.
  '''
  #If data alrdy in approp. numeric format, you can pack data into a vec b4 passing to model
  SELECT_COLS = ['survived', 'age', 'n_siblings_spouses', 'parch', 'fare']
  DEFAULTS = [0, 0.0, 0.0, 0.0, 0.0]
  temp_dataset = get_dataset(
    COL_LABEL,
    train_file_path, 
    select_columns=SELECT_COLS,
    column_defaults = DEFAULTS)

  show_batch(temp_dataset)
  example_batch, labels_batch = next(iter(temp_dataset))
  packed_dataset = temp_dataset.map(pack)

  print('\n')
  for features, labels in packed_dataset.take(1):
    print(features.numpy())
    print()
    print(labels.numpy())
  
  example_batch, labels_batch = next(iter(temp_dataset)) # (2)calling again need to understand why better?
  
  NUMERIC_FEATURES = ['age','n_siblings_spouses','parch', 'fare']
  packed_train_data = raw_train_data.map(PackNumericFeatures(NUMERIC_FEATURES))
  packed_test_data = raw_test_data.map(PackNumericFeatures(NUMERIC_FEATURES))
  show_batch(packed_train_data)

  example_batch, labels_batch = next(iter(temp_dataset)) # (3)calling again need to understand why better?

  # Now the continuous data needs to be normalized w/ pandas


'''
  Helper Functions 
  (not modular in my Tf2KerasRef/*'s)
  These are MVP's
'''
# Each item in the dataset is a batch, represented as a tuple. {:20s} is python for 20 spaces :after
def show_batch(_dataset):
  print('\n')
  for batch, label in _dataset.take(1):
    for key, value in batch.items():
      print("{:20s}: {}".format(key,value.numpy()))

# For reading csv data and creating dataset. col_label is my label I am testing
def get_dataset(_col_label, _path, **kwargs):
  new_dataset = tf.data.experimental.make_csv_dataset(
    _path,
    batch_size=5,
    label_name=_col_label,
    na_value="?",
    num_epochs=1,
    ignore_errors=True,
    **kwargs
  )
  return new_dataset

# Packs together all cols
def pack(_features, _label):
  return tf.stack(list(_features.values()), axis=-1), _label

# general preprocessor that selects a list of numeric feats and packs them into a single col
class PackNumericFeatures(object):
  def __init__(self, _names):
    self.names = _names
  def __call__(self, _features, _labels):
    numeric_features = [_features.pop(name) for name in self.names]
    numeric_features = [tf.cast(feat, tf.float32) for feat in numeric_features]
    numeric_features = tf.stack(numeric_features, axis=-1)
    _features['numeric'] = numeric_features
    return _features, _labels


if __name__ == '__main__':
  main()
