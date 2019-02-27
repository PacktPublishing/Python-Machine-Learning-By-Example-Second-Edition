'''
Source codes for Python Machine Learning By Example 2nd Edition (Packt Publishing)
Chapter 5: Classifying Newsgroup Topic with Support Vector Machine
Author: Yuxi (Hayden) Liu
'''

import tensorflow as tf
import numpy as np

from sklearn import datasets
cancer_data = datasets.load_breast_cancer()
X = cancer_data.data
Y = cancer_data.target

from collections import Counter
print(Counter(Y))


np.random.seed(42)
train_indices = np.random.choice(len(Y), round(len(Y) * 0.8), replace=False)
test_indices = np.array(list(set(range(len(Y))) - set(train_indices)))
X_train = X[train_indices]
X_test = X[test_indices]
Y_train = Y[train_indices]
Y_test = Y[test_indices]


svm_tf = tf.contrib.learn.SVM(
  feature_columns=(tf.contrib.layers.real_valued_column(column_name='x'),),
  example_id_column='example_id')



input_fn_train = tf.estimator.inputs.numpy_input_fn(
    x={'x': X_train, 'example_id': np.array(['%d' % i for i in range(len(Y_train))])},
    y=Y_train,
    num_epochs=None,
    batch_size=100,
    shuffle=True)



svm_tf.fit(input_fn=input_fn_train, max_steps=100)


metrics = svm_tf.evaluate(input_fn=input_fn_train, steps=1)
print('The training accuracy is: {0:.1f}%'.format(metrics['accuracy']*100))



input_fn_test = tf.estimator.inputs.numpy_input_fn(
    x={'x': X_test, 'example_id': np.array(['%d' % (i + len(Y_train)) for i in range(len(X_test))])},
    y=Y_test,
    num_epochs=None,
    shuffle=False)


metrics = svm_tf.evaluate(input_fn=input_fn_test, steps=1)
print('The testing accuracy is: {0:.1f}%'.format(metrics['accuracy']*100))

