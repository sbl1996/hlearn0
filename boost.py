from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.base import clone

class AdaBooster:

  def __init__(self, base_estimator, n_estimators):
    self.base_estimator = base_estimator
    self.n_estimators = n_estimators

  def fit(self, X, y):
    T = self.n_estimators
    m = len(X)
    w = np.repeat(1.0 / m, m)
    alpha = np.zeros((T,))
    estimators = []
    for t in range(T):
      ind = np.random.choice(np.arange(m), size=m, p=w)
      Xb = X[ind]
      yb = y[ind]
      f = clone(self.base_estimator).fit(Xb, yb)
      estimators.append(f)
      pred = f.predict(X)
      eps = w @ (pred != y)
      alpha[t] = np.log((1 - eps) / (eps + 1e-8)) / 2
      w = w * np.exp(-alpha[t] * y * pred)
      w = w / w.sum()
    self.estimators = estimators
    self.alpha = alpha

    return self

  def predict(self, X):
    m = len(X)
    res = np.zeros((m,))
    for t in range(self.n_estimators):
      res += self.alpha[t] * self.estimators[t].predict(X)
    res[res == 0] = 1
    return np.sign(res)

  def score(self, X, y):
    return accuracy_score(y, self.predict(X))

  def get_params(self, deep=False):
    return {
      'base_estimator': self.base_estimator,
      'n_estimators': self.n_estimators,
    }


bre = datasets.load_breast_cancer()
X = bre.data
y = bre.target
y[y == 0] = -1
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

clf = DecisionTreeClassifier()
cross_val_score(clf, X, y, cv=5)

clf = RandomForestClassifier(20)
cross_val_score(clf, X, y, cv=5)

ada = AdaBoostClassifier(DecisionTreeClassifier(), 1000)
cross_val_score(ada, X, y, cv=5)

bst = AdaBooster(DecisionTreeClassifier(max_depth=1), 20)
bst.fit(X_train, y_train)
accuracy_score(y_test, bst.predict(X_test))