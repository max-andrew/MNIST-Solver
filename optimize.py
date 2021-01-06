from sklearn import neural_network
from sklearn import preprocessing
from sklearn.datasets import fetch_mldata
from sklearn.model_selection import cross_val_score

mnist = fetch_mldata('MNIST original')
mnist_scaled = preprocessing.StandardScaler(mnist.data)

# classifiers
mlpc = neural_network.MLPClassifier(hidden_layer_sizes=(1), solver='lbfgs')
mlpc4 = neural_network.MLPClassifier(hidden_layer_sizes=(4), solver='lbfgs')
mlpc6 = neural_network.MLPClassifier(hidden_layer_sizes=(6), solver='lbfgs')
mlpc8 = neural_network.MLPClassifier(hidden_layer_sizes=(8), solver='lbfgs')

# accuracies
score = cross_val_score(mlpc, mnist.data, mnist.target, cv=4)
score4 = cross_val_score(mlpc4, mnist.data, mnist.target, cv=4)
score6 = cross_val_score(mlpc6, mnist.data, mnist.target, cv=4)
score8 = cross_val_score(mlpc8, mnist.data, mnist.target, cv=4)

# scaled accuracies
scale_score = cross_val_score(mlpc, mnist_scaled, mnist.target, cv=4)
scale_score4 = cross_val_score(mlpc4, mnist_scaled, mnist.target, cv=4)
scale_score6 = cross_val_score(mlpc6, mnist_scaled, mnist.target, cv=4)
scale_score8 = cross_val_score(mlpc8, mnist_scaled, mnist.target, cv=4)

# print mean classification accuracies
print score.mean()
print score4.mean()
print score6.mean()
print score8.mean()

print scale_score.mean()
print scale_score4.mean()
print scale_score6.mean()
print scale_score8.mean()