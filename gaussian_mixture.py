from scipy.stats import multivariate_normal
from sklearn.cluster import KMeans

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

def gaussian_mixture(X, K, tol=1e-4, max_iter=100):
  m, n = X.shape
  km = KMeans(K).fit(X)
  cnt = np.unique(km.labels_, return_counts=True)[1]
  p = cnt / cnt.sum()
  mean = km.cluster_centers_
  cov = np.empty((K, n, n), dtype=np.float64)
  for k in range(K):
    cov[k] = np.cov(X[km.labels_ == k], rowvar=False)

  weight = np.empty((m, K), dtype=np.float64)

  L = -np.inf
  it = 0
  while it < max_iter:

    for k in range(K):
      weight[:, k] = p[k] * multivariate_normal.pdf(X, mean[k], cov[k], allow_singular=True)
    weight = weight / weight.sum(axis=1, keepdims=True)

    for k in range(K):
      w = weight[:, k].sum()
      p[k] = w / m
      mean[k] = np.sum(weight[:, [k]] * X, axis=0) / w
      diff = X - mean[k]
      cov[k] = diff.T @ (diff * weight[:, [k]]) / w
    
    L0 = L
    L = 0
    for k in range(K):
      log_prob = multivariate_normal.logpdf(X, mean[k], cov[k], allow_singular=True)
      L += np.sum((log_prob + np.log(p[k])) * weight[:, k])
    print('Iter %d: %f' % (it, L))
    if L - L0 < tol:
      break
    it += 1
  return p, mean, cov, weight