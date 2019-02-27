'''
Source codes for Python Machine Learning By Example 2nd Edition (Packt Publishing)
Chapter 6: Predicting Online Ads Click-through with Tree-Based Algorithms
Author: Yuxi (Hayden) Liu
'''

import pandas as pd
n_rows = 300000
df = pd.read_csv("train", nrows=n_rows)
print(df.head(5))


X = df.drop(['click', 'id', 'hour', 'device_id', 'device_ip'], axis=1).values
Y = df['click'].values

print(X.shape)

n_train = int(n_rows * 0.9)
X_train = X[:n_train]
Y_train = Y[:n_train]
X_test = X[n_train:]
Y_test = Y[n_train:]

from sklearn.preprocessing import OneHotEncoder
enc = OneHotEncoder(handle_unknown='ignore')
X_train_enc = enc.fit_transform(X_train)

X_train_enc[0]
print(X_train_enc[0])


X_test_enc = enc.transform(X_test)


import tensorflow as tf
from tensorflow.contrib.tensor_forest.python import tensor_forest
from tensorflow.python.ops import resources


n_iter = 20
n_classes = 2
n_features = int(X_train_enc.toarray().shape[1])
n_trees = 10
max_nodes = 30000


x = tf.placeholder(tf.float32, shape=[None, n_features])
y = tf.placeholder(tf.int64, shape=[None])


hparams = tensor_forest.ForestHParams(num_classes=n_classes, num_features=n_features, num_trees=n_trees,
                                      max_nodes=max_nodes, split_after_samples=30).fill()

forest_graph = tensor_forest.RandomForestGraphs(hparams)

train_op = forest_graph.training_graph(x, y)
loss_op = forest_graph.training_loss(x, y)

infer_op, _, _ = forest_graph.inference_graph(x)

auc = tf.metrics.auc(tf.cast(y, tf.int64), infer_op[:, 1])[1]


init_vars = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer(), resources.initialize_resources(resources.shared_resources()))

sess = tf.Session()

sess.run(init_vars)

batch_size = 1000

import numpy as np
indices = list(range(n_train))

def gen_batch(indices):
    np.random.shuffle(indices)
    for batch_i in range(int(n_train / batch_size)):
        batch_index = indices[batch_i*batch_size: (batch_i+1)*batch_size]
        yield X_train_enc[batch_index], Y_train[batch_index]


for i in range(1, n_iter + 1):
    for X_batch, Y_batch in gen_batch(indices):
        _, l = sess.run([train_op, loss_op], feed_dict={x: X_batch.toarray(), y: Y_batch})
    acc_train = sess.run(auc, feed_dict={x: X_train_enc.toarray(), y: Y_train})
    print('Iteration %i, AUC of ROC on training set: %f' % (i, acc_train))
    acc_test = sess.run(auc, feed_dict={x: X_test_enc.toarray(), y: Y_test})
    print("AUC of ROC on testing set:", acc_test)

