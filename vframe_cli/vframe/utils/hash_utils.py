import hashlib

HASH_TREE_DEPTH = 2
HASH_BRANCH_SIZE = 2

def sha256(fp_in, block_size=65536):
  """Generates SHA256 hash for a file
  :param fp_in: (str) filepath
  :param block_size: (int) byte size of block
  :returns: (str) hash
  """
  sha = hashlib.sha256()
  with open(fp_in, 'rb') as fp:
    for block in iter(lambda: fp.read(block_size), b''):
      sha.update(block)
  return sha.hexdigest()

def sha256_stream(stream, block_size=65536):
  """Generates SHA256 hash for a file stream (from Flask)
  :param fp_in: (FileStream) stream object
  :param block_size: (int) byte size of block
  :returns: (str) hash
  """
  sha = hashlib.sha256()
  for block in iter(lambda: stream.read(block_size), b''):
    sha.update(block)
  return sha.hexdigest()

def sha256_tree(sha, depth=HASH_TREE_DEPTH, branch_size=HASH_BRANCH_SIZE):
  """Split hash into branches with tree-depth for faster file indexing
  :param sha: str of a sha256 hash
  :returns: str with sha256 tree with '/' delimeter
  """
  branch_size = HASH_BRANCH_SIZE
  tree_size = HASH_TREE_DEPTH * branch_size
  tree = [sha[i:(i+branch_size)] for i in range(0, tree_size, branch_size)]
  return '/'.join(tree)
