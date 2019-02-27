'''
Source codes for Python Machine Learning By Example 2nd Edition (Packt Publishing)
Chapter 9: Stock Price Prediction with Regression Algorithms
Author: Yuxi (Hayden) Liu
'''
import numpy as np


def sigmoid(z):
    return 1.0 / (1 + np.exp(-z))


def sigmoid_derivative(z):
    return sigmoid(z) * (1.0 - sigmoid(z))



def train(X, y, n_hidden, learning_rate, n_iter):
    m, n_input = X.shape
    W1 = np.random.randn(n_input, n_hidden)
    b1 = np.zeros((1, n_hidden))
    W2 = np.random.randn(n_hidden, 1)
    b2 = np.zeros((1, 1))
    for i in range(1, n_iter+1):
        Z2 = np.matmul(X, W1) + b1
        A2 = sigmoid(Z2)
        Z3 = np.matmul(A2, W2) + b2
        A3 = Z3

        dZ3 = A3 - y
        dW2 = np.matmul(A2.T, dZ3)
        db2 = np.sum(dZ3, axis=0, keepdims=True)

        dZ2 = np.matmul(dZ3, W2.T) * sigmoid_derivative(Z2)
        dW1 = np.matmul(X.T, dZ2)
        db1 = np.sum(dZ2, axis=0)

        W2 = W2 - learning_rate * dW2 / m
        b2 = b2 - learning_rate * db2 / m
        W1 = W1 - learning_rate * dW1 / m
        b1 = b1 - learning_rate * db1 / m

        if i % 100 == 0:
            cost = np.mean((y - A3) ** 2)
            print('Iteration %i, training loss: %f' % (i, cost))

    model = {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}
    return model


def predict(x, model):
    W1 = model['W1']
    b1 = model['b1']
    W2 = model['W2']
    b2 = model['b2']
    A2 = sigmoid(np.matmul(x, W1) + b1)
    A3 = np.matmul(A2, W2) + b2
    return A3



from sklearn import datasets
boston = datasets.load_boston()
num_test = 10  # the last 10 samples as testing set

from sklearn import preprocessing
scaler = preprocessing.StandardScaler()

X_train = boston.data[:-num_test, :]
X_train = scaler.fit_transform(X_train)
y_train = boston.target[:-num_test].reshape(-1, 1)
X_test = boston.data[-num_test:, :]
X_test = scaler.transform(X_test)
y_test = boston.target[-num_test:]


n_hidden = 20
learning_rate = 0.1
n_iter = 2000

model = train(X_train, y_train, n_hidden, learning_rate, n_iter)
predictions = predict(X_test, model)
print(predictions)
print(y_test)




# Scikit-learn implementation of neural network

from sklearn.neural_network import MLPRegressor
nn_scikit = MLPRegressor(hidden_layer_sizes=(20, 8), activation='logistic', solver='lbfgs',
                         learning_rate_init=0.1, random_state=42, max_iter=2000)
nn_scikit.fit(X_train, y_train)
predictions = nn_scikit.predict(X_test)
print(predictions)
print(np.mean((y_test - predictions) ** 2))


# TensorFlow implementation of neural network

import tensorflow as tf
n_features = int(X_train.shape[1])
n_hidden_1 = 20
n_hidden_2 = 8

learning_rate = 0.1
n_iter = 2000

x = tf.placeholder(tf.float32, shape=[None, n_features])
y = tf.placeholder(tf.float32, shape=[None, 1])

layer_1 = tf.nn.sigmoid(tf.layers.dense(x, n_hidden_1))
layer_2 = tf.nn.sigmoid(tf.layers.dense(layer_1, n_hidden_2))
pred = tf.layers.dense(layer_2, 1)


cost = tf.losses.mean_squared_error(labels=y, predictions=pred)
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

init_vars = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init_vars)

for i in range(1, n_iter+1):
    _, c = sess.run([optimizer, cost], feed_dict={x: X_train, y: y_train})
    if i % 100 == 0:
        print('Iteration %i, training loss: %f' % (i, c))

predictions = sess.run(pred, feed_dict={x: X_test})
print(predictions)




# Keras implementation of neural network


from keras import models
from keras import layers

model = models.Sequential()
model.add(layers.Dense(n_hidden_1, activation="sigmoid", input_shape=(n_features, )))
model.add(layers.Dense(n_hidden_2, activation="sigmoid"))
model.add(layers.Dense(1))


from keras import optimizers
sgd = optimizers.SGD(lr=0.01)
model.compile(loss='mean_squared_error', optimizer=sgd)


model.fit(
    X_train, y_train,
    epochs=100,
    validation_data=(X_test, y_test)
)

predictions = model.predict(X_test)
print(predictions)
