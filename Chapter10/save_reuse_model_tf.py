'''
Source codes for Python Machine Learning By Example 2nd Edition (Packt Publishing)
Chapter 10: Machine Learning Best Practices
Author: Yuxi (Hayden) Liu
'''

import tensorflow as tf


from sklearn import datasets
cancer_data = datasets.load_breast_cancer()
X = cancer_data.data
Y = cancer_data.target

n_features = int(X.shape[1])
learning_rate = 0.005
n_iter = 200


# Input and Target placeholders
x = tf.placeholder(tf.float32, shape=[None, n_features])
y = tf.placeholder(tf.float32, shape=[None])

# Build the logistic regression model
W = tf.Variable(tf.zeros([n_features, 1]), name='W')
b = tf.Variable(tf.zeros([1]), name='b')

logits = tf.add(tf.matmul(x, W), b)[:, 0]
cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=logits))

optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

sess = tf.Session()

sess.run(tf.global_variables_initializer())

for i in range(1, n_iter+1):
    _, c = sess.run([optimizer, cost], feed_dict={x: X, y: Y})
    if i % 10 == 0:
        print('Iteration %i, training loss: %f' % (i, c))

# Save the trained model
# create saver object
saver = tf.train.Saver()

file_path = './model_tf'
saved_path = saver.save(sess, file_path)
print('model saved in path: {}'.format(saved_path))


tf.reset_default_graph()

# Load the graph from the file
imported_graph = tf.train.import_meta_graph(file_path+'.meta')



with tf.Session() as sess:
    # restore the saved model
    imported_graph.restore(sess, file_path)
    # print the loaded weights
    W_loaded, b_loaded = sess.run(['W:0','b:0'])
    print('Saved W = ', W_loaded)
    print('Saved b = ', b_loaded)

