'''
Source codes for Python Machine Learning By Example 2nd Edition (Packt Publishing)
Chapter 2: Exploring the 20 Newsgroups Dataset with Text Analysis Techniques
Author: Yuxi (Hayden) Liu
'''


from sklearn.datasets import fetch_20newsgroups


groups = fetch_20newsgroups()
groups.keys()
groups['target_names']
groups.target


import numpy as np
np.unique(groups.target)



import seaborn as sns
sns.distplot(groups.target)
import matplotlib.pyplot as plt
plt.show()


groups.data[0]
groups.target[0]
groups.target_names[groups.target[0]]



