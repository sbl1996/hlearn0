import numpy as np

def gen_func(M):
  def func(X, w):
    return X @ w
  return func

def poly(x, M):
  num_examples = len(x)
  X = np.zeros((num_examples, M))
  for i in range(num_examples):
    for j in range(M):
      X[i, j] = x[i] ** (j + 1)
  return X

def pad_ones(X):
  return np.c_[np.ones((len(X),)), X]

def rfunc(x):
  return np.sin(2 * np.pi * x)

def plot(rfunc, start, end):
  x = np.linspace(start, end, 200)
  y = rfunc(x)
  plt.plot(x, y)
  plt.pause(0.1)

def cost_func(X, y, w, func):
  pred = func(X, w)
  cost = np.sum((pred - y) ** 2) / 2
  grad = X.T @ (pred - y)
  return cost, grad

def gradient_descent(fun, x0, learning_rate=0.01, num_iter=50, tol=0.0001):
  x = x0
  for i in range(num_iter):
    cost, grad = fun(x)
    if cost < tol:
      return x, cost
    x = x - learning_rate * grad
  cost, grad = fun(x)
  return x, cost

def momentum(fun, x0, learning_rate=0.01, momentum=0.9, num_iter=50):
  x = x0
  m = np.zeros(x.shape)
  for i in range(num_iter):
    cost, grad = fun(x)
    m = momentum * m + learning_rate * grad
    x = x - m
  cost, grad = fun(x)
  return x, cost  

def ada_grad(fun, x0, learning_rate=0.01, epsilon=1e-10, num_iter=50):
  x = x0
  s = np.zeros(x.shape)
  for i in range(num_iter):
    cost, grad = fun(x)
    s = s + grad ** 2
    x = x - learning_rate * (grad / np.sqrt(s + epsilon))
  cost, grad = fun(x)
  return x, cost

def rmsprop(fun, x0, learning_rate=0.01, momentum=0.9, decay=0.9, epsilon=1e-10, num_iter=50):
  x = x0
  s = np.zeros(x.shape)
  for i in range(num_iter):
    cost, grad = fun(x)
    s = momentum * s + (1 - decay) * (grad ** 2)
    x = x - learning_rate * grad / np.sqrt(s + epsilon)
  cost, grad = fun(x)
  return x, cost

def adam(fun, x0, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8, num_iter=50):
  x = x0
  m = np.zeros(x.shape)
  s = np.zeros(x.shape)
  for i in range(num_iter):
    cost, grad = fun(x)
    m = beta1 * m + (1 - beta1) * grad
    s = beta2 * s + (1 - beta2) * (grad ** 2)
    m = m / (1 - beta1 ** (i + 1))
    s = s / (1 - beta2 ** (i + 1))
    x = x - learning_rate * m / np.sqrt(s + epsilon)
  cost, grad = fun(x)
  return x, cost

def rms(model, x, y):
  pred = model.predict(x)
  cost = np.sum((pred - y) ** 2) / 2
  return np.sqrt((2 * cost) / len(x))

def find_optimizer(name):
  optimizers = {
    'momentum': momentum,
    'gd': gradient_descent,
    'adagrad': ada_grad,
    'rmsprop': rmsprop,
    'adam': adam
  }
  return optimizers[name]

class LinearRegression(object):

  def __init__(self, M):
    self.M = M
    self.func = gen_func(M)
    self.w = np.random.randn(M + 1)
    self.w[0] = 0

  def fit(self, x, y, optimizer='gd', **kwargs):
    X = self._preprocess(x)
    w0 = self.w

    if isinstance(optimizer, str):
      optimizer = find_optimizer(optimizer)
    w, _ = optimizer(lambda w: cost_func(X, y, w, self.func), w0, **kwargs)
    self.w = w

  def _preprocess(self, x):
    X = poly(x, self.M)
    X = pad_ones(X)
    return X

  def predict(self, x):
    X = self._preprocess(x)
    return self.func(X, self.w)

  def evaluate(self, x, y):
    pred = model.predict(x)
    return np.sum((pred - y) ** 2) / 2

model = LinearRegression(3)
model.fit(x1, y1, learning_rate=0.01, num_iter=100000)
model.fit(x1, y1, optimizer='momentum', learning_rate=0.01, num_iter=100000)
model.fit(x1, y1, optimizer='rmsprop', learning_rate=0.01, num_iter=100000)
model.fit(x1, y1, optimizer='adam', learning_rate=0.01, num_iter=100000)
x = np.random.uniform(0, 1, 10)
y = rfunc(x)