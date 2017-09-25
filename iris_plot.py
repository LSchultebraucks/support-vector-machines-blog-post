import matplotlib.pyplot as plt
from sklearn import datasets

iris = datasets.load_iris()
features = iris.data[:, :2]  # only taking first two features
labels = iris.target

plt.scatter(features[:, 0], features[:, 1], c=labels, cmap=plt.cm.Set1)
plt.plot([4.4, 6.5], [2.1, 4.3])
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')

plt.xticks(())
plt.yticks(())

plt.legend()

plt.show()