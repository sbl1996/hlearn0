import numpy as np

def k_init(X, K):
  m, n = X.shape
  centers = np.zeros((K,n))
  centers[0] = X[np.random.randint(m)]
  for i in range(1, K):
    d = np.repeat(np.inf, m)
    for j in range(i):
      d = np.minimum(d, distance(X, centers[j]))
    p = d / d.sum()
    ind = np.random.choice(np.arange(m), p=p)
    centers[i] = X[ind]
  return centers


def distance(X, p):
  return ((X - p) ** 2).sum(axis=-1)

def kmeans(X, K, tol=1e-4, max_iter=300):
  m = len(X)
  mu = k_init(X, K)
  L = np.inf
  it = 0
  label = np.zeros((m,), dtype=np.int)
  while it < max_iter:
    for i in range(m):
      label[i] = np.argmin(distance(mu, X[i]))
    for k in range(K):
      mu[k] = X[label == k].mean(axis=0)
    L0 = L
    L = 0
    for i in range(m):
      L += distance(mu[label[i]], X[i])
    print('Iter %d: %f' % (it, L))
    if abs(L - L0) < tol:
      break
    it += 1
  return mu, label

def weighted_kmeans(X, K, tol=1e-4, max_iter=100, beta=1):
  m = len(X)
  ind = np.random.choice(m, size=K, replace=False)
  mu = X[ind]
  L = np.inf
  it = 0
  weight = np.zeros((m,K))
  while it < max_iter:
    for i in range(m):
      d = distance(mu, X[i])
      weight[i] = np.exp(-d / beta)
      weight[i] = weight[i] / weight[i].sum()
    for k in range(K):
      mu[k] = (X * weight[:, [k]]).sum(axis=0) / weight[:, k].sum()
    L0 = L
    L = 0
    for k in range(K):
      L += (weight[:, k] * distance(X, mu[k]) / beta).sum()
    print('Iter %d: %f' % (it, L))
    if abs(L - L0) < tol:
      break
    it += 1
  return mu, weight