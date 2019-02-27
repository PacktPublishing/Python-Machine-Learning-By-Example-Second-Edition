import tensorflow as tf


import pandas as pd
n_rows = 300000
df = pd.read_csv("train", nrows=n_rows)

X = df.drop(['click', 'id', 'hour', 'device_id', 'device_ip'], axis=1).values
Y = df['click'].values

n_train = int(n_rows * 0.9)
X_train = X[:n_train]
Y_train = Y[:n_train]
X_test = X[n_train:]
Y_test = Y[n_train:]

from sklearn.preprocessing import OneHotEncoder
enc = OneHotEncoder(handle_unknown='ignore')
X_train_enc = enc.fit_transform(X_train)
X_test_enc = enc.transform(X_test)


n_features = int(X_train_enc.toarray().shape[1])
learning_rate = 0.001
n_iter = 20


# Input and Target placeholders
x = tf.placeholder(tf.float32, shape=[None, n_features])
y = tf.placeholder(tf.float32, shape=[None])

# Build the logistic regression model
W = tf.Variable(tf.zeros([n_features, 1]))
b = tf.Variable(tf.zeros([1]))

logits = tf.add(tf.matmul(x, W), b)[:, 0]
pred = tf.nn.sigmoid(logits)

cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=logits))
auc = tf.metrics.auc(tf.cast(y, tf.int64), pred)[1]


optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)



# Initialize the variables (i.e. assign their default value)
init_vars = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())


batch_size = 1000

import numpy as np
indices = list(range(n_train))

def gen_batch(indices):
    np.random.shuffle(indices)
    for batch_i in range(int(n_train / batch_size)):
        batch_index = indices[batch_i*batch_size: (batch_i+1)*batch_size]
        yield X_train_enc[batch_index], Y_train[batch_index]


sess = tf.Session()

sess.run(init_vars)


for i in range(1, n_iter+1):
    avg_cost = 0.
    for X_batch, Y_batch in gen_batch(indices):
        _, c = sess.run([optimizer, cost], feed_dict={x: X_batch.toarray(), y: Y_batch})
        avg_cost += c / int(n_train / batch_size)
    print('Iteration %i, training loss: %f' % (i, avg_cost))


auc_test = sess.run(auc, feed_dict={x: X_test_enc.toarray(), y: Y_test})
print("AUC of ROC on testing set:", auc_test)
