import numpy as np
from collections import namedtuple

class Node():
  def __init__(self, data, filter=None, left=None, right=None):
    self.filter = filter
    self.data = data
    self.left = left
    self.right = right

NodeData = namedtuple('NodeData', ['gini', 'samples', 'value', 'label'])

class DecisionTreeClassifier():
  def __init__(self, max_depth=2):
    self.max_depth = max_depth

  def fit(self, X, y):
    m, n = X.shape
    classes, counts = np.unique(y, return_counts=True)
    self.num_classes = len(classes)
    counts_root = np.repeat(0, self.num_classes)
    counts_root[classes] = counts
    gini_root = self.gini(counts_root)
    label_root = np.argmax(gini_root)
    data = NodeData(gini_root, m, counts_root, label_root)
    self.root = self.make_tree(X, y, data, self.max_depth)

  def make_tree(self, X, y, data, depth):
    tree = Node(data=data)
    if depth == 0:
      return tree
    cost, k, threshold, data_left, data_right = self.best_split(X, y)
    tree.filter = (k, threshold)

    if cost != 0:
      ind = X[:, k] <= threshold
      if data_left.gini == 0:
        tree.left = Node(data_left)
        tree.right = self.make_tree(X[~ind], y[~ind], data_right, depth - 1)
      elif data_right.gini == 0:
        tree.left = self.make_tree(X[ind], y[ind], data_left, depth - 1)
        tree.right = Node(data_right)
      else:
        tree.left = self.make_tree(X[ind], y[ind], data_left, depth - 1)
        tree.right = self.make_tree(X[~ind], y[~ind], data_right, depth - 1)
    else:
      tree.left = Node(data_left)
      tree.right = Node(data_right)
    return tree

  def gini(self, counts):
    s = np.sum(counts)
    if s == 0:
      return 1
    p = counts / s
    return 1 - np.sum(p ** 2)

  def get_data(self, X, y, mask):
    y = y[mask]
    m = len(y)
    classes, counts = np.unique(y, return_counts=True)
    value = np.repeat(0, self.num_classes)
    value[classes] = counts
    gini = self.gini(value)
    label = np.argmax(value)
    return NodeData(gini, m, value, label)

  def best_split(self, X, y):
      m, n = X.shape
      min_cost = np.inf
      best = None
      for k in range(n):
        thresholds = np.unique(X[:, k])
        for threshold in thresholds:
          mask = X[:, k] <= threshold

          data_left = self.get_data(X, y, mask)
          cost_left = (data_left.samples / m) * data_left.gini

          data_right = self.get_data(X, y, ~mask)
          cost_right = (data_right.samples / m) * data_right.gini

          cost = cost_left + cost_right
          if cost < min_cost:
            min_cost = cost
            best = (cost, k, threshold, data_left, data_right)
      return best

  def predict(self, X):
    m = len(X)
    y = np.zeros((m,))
    for i in range(m):
      tree = self.root
      while tree:
        if tree.filter:
          k, threshold = tree.filter
          if X[i, k] <= threshold:
            tree = tree.left
          else:
            tree = tree.right
        else:
          y[i] = tree.data.label
          break
    return y
