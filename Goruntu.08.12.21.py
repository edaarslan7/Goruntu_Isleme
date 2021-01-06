#!/usr/bin/env python
# coding: utf-8

%matplotlib inline
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from sklearn.datasets import fetch_openml
dataset = fetch_openml("mnist_784")

#type(dataset), type(dataset["data"]), type(dataset["data"])

X, y = dataset["data"], dataset["target"]

X = X/255

#X.shape

#y.shape

img1 = X[1,:].reshape(28,28)

plt.imshow(img_1,cmap='gray')
plt.show()

#y[1]

i=100
img1 = X[100,:].reshape(28,28)
plt.imshow(img1,cmap='gray')
plt.title(y[i])
plt.show()

y_new = np.zeros(y.shape)
y_new[np.where(y == 0.0)[0]] = 1
y = y_new
#y

m = 60000
m_test = X.shape[0] - m

X_train, X_test = X[:m].T, X[m:].T
y_train, y_test = y[:m].reshape(1,m), y[m:].reshape(1,m_test)

np.random.seed(138)
shuffle_index = np.random.permutation(m)
X_train, y_train = X_train[:,shuffle_index], y_train[:,shuffle_index]

i = 3
plt.imshow(X_train[:,1].reshape(28,28), cmap = matplotlib.cm.binary)
plt.axis("off")
plt.show()
print(y_train[:,i])

epsilon = 1e-5
def sigmoid(z):
    s = 1 / (1 + np.exp(-z))
    return s
    
def compute_loss(Y, Y_hat):
    m = Y.shape[1]
    l = -(1./m) * (np.sum(np.multiply(np.log(Y_hat+epsilon),Y)) + np.sum(np.multiply(np.log(1-Y_hat+epsilon),(1-Y))))
    return l
    
learning_rate = 1
X = X_train
Y = y_train

n_x = X.shape[0]
m = X.shape[1]

W = np.random.randn(n_x, 1) * 0.01
b = np.zeros((1,1))
for i in range(2000):
    Z = np.matmul(W.T, X) + b
    A = sigmoid(Z)
    cost = compute_loss(Y,A)
    
    dW = (1/m)*np.matmul(X, (A-Y).T)
    db = (1/m)*np.sum(A-Y, axis=1, keepdims=True)
    
    W = W - learning_rate * dW
    b = b - learning_rate * db
    
    if(i%100==0):
        print("Epoch", i, "cost: ", cost)
print("final cost: ", cost)

y_test[0,100] = 1.0

from sklearn.metrics import classification_report, confusion_matrix

Z = np.matmul(W.T, X_test) + b
A = sigmoid(Z)

predictions = (A>.5)[0,:]
labels = (y_test == 1)[0,:]

print(confusion_matrix(predictions, labels))

print(classification_report(predictions, labels))
