def SMO(X, y, K, C, tol, max_passes):
  m = len(X)
  alpha = np.zeros((m,))
  b = 0
  passes = 0
  while passes < max_passes:
    num_changed_alphas = 0
    for i in range(m):
      E_i = alpha * y @ K[i] + b - y[i]
      if (y[i] * E_i < -tol and alpha[i] < C) or (y[i] * E_i > tol and alpha[i] > 0):
        j = np.random.choice(np.setdiff1d(np.arange(m), i))
        E_j = alpha * y @ K[j] + b - y[j]
        alpha_i_old = alpha[i]
        alpha_j_old = alpha[j]
        if y[i] != y[j]:
          L = max(0, alpha[j] - alpha[i])
          H = min(C, C + alpha[j] - alpha[i])
        else:
          L = max(0, alpha[i] + alpha[j] - C)
          H = min(C, alpha[i] + alpha[j])
        if L == H:
          continue
        eta = 2 * K[i, j] - K[i, i] - K[j, j]
        if eta >= 0:
          continue
        alpha[j] = np.clip(alpha[j] - y[j] * (E_i - E_j) / eta, L, H)
        if np.abs(alpha[j] - alpha_j_old) < 1e-5:
          continue
        alpha[i] = alpha[i] + y[i] * y[j] * (alpha_j_old - alpha[j])
        b1 = b - E_i - y[i] * (alpha[i] - alpha_i_old) * K[i, i] - y[j] * (alpha[j] - alpha_j_old) * K[i, j]
        b2 = b - E_j - y[i] * (alpha[i] - alpha_i_old) * K[i, j] - y[j] * (alpha[j] - alpha_j_old) * K[j, j]
        if 0 < alpha[i] < C:
          b = b1
        elif 0 < alpha[j] < C:
          b = b2
        else:
          b = (b1 + b2) / 2
        num_changed_alphas = num_changed_alphas + 1
    if num_changed_alphas == 0:
      passes = passes + 1
    else:
      passes = 0
  return alpha, b

class SVM():
  
  def __init__(self, kernel='linear', C=1.0, tol=0.001, max_passes=50):
    super(SVM, self).__init__()
    self.kernel = kernel
    self.C = C
    self.tol = tol
    self.max_passes = max_passes

  def fit(self, X, y):
    K = X @ X.T
    C = self.C
    tol = self.tol
    max_passes = self.max_passes
    alpha, b = SMO(X, y, K, C, tol, max_passes)
    self.alpha = alpha
    self.b = b
    self.w = np.sum((alpha * y).reshape(-1, 1) * X, axis=0)
    self.support_ = np.nonzero(self.alpha > 1e-10)[0]
    return self

  def decision_function(self, X):
    return X @ self.w + self.b

  def predict(self, X):
    y = self.decision_function(X)
    y[y >= 0] = 1
    y[y < 0] = -1
    return y  