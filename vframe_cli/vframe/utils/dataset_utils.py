import random

def split_train_val_test(annos, randomize=True, splits=(0.6, 0.2, 0.2), seed=1):
  """Convert annotation index into splits based on classes
  :param annos: (list) of filepaths
  :param randomize: (bool) randomize list
  :param splits: (tuple) splits of train, val, test
  :param seed: (int) random seed
  :returns (dict) of train, val, test lists
  """

  if randomize:
    random.seed(seed)
    random.shuffle(annos)

  n_annos = len(annos)
  n_train, n_val, n_test = splits
  n_train = int(n_annos * n_train)
  n_val = int(n_annos * n_val)
  result = {
    'train': annos[:n_train],
    'val': annos[n_train:n_train + n_val],
    'test': annos[n_train + n_val:],
  }
  return result
