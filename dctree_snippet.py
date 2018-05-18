from decision_tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score

iris = load_iris()
X = iris.data
y = iris.target

dc1 = DecisionTreeClassifier(max_depth=2)
dc1.fit(X, y)
y1 = dc1.predict(X)
accuracy_score(y, y1)

t = dc1.root