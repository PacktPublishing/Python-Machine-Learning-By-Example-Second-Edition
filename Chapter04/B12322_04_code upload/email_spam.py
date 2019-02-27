'''
Source codes for Python Machine Learning By Example 2nd Edition (Packt Publishing)
Chapter 4: Detecting Spam Email with Naive Bayes
Author: Yuxi (Hayden) Liu
'''

# -*- coding: utf-8 -*-

import glob
import os
import numpy as np


file_path = 'enron1/ham/0007.1999-12-14.farmer.ham.txt'
with open(file_path, 'r') as infile:
    ham_sample = infile.read()

print(ham_sample)

file_path = 'enron1/spam/0058.2003-12-21.GP.spam.txt'
with open(file_path, 'r') as infile:
    spam_sample = infile.read()

print(spam_sample)


emails, labels = [], []

file_path = 'enron1/spam/'
for filename in glob.glob(os.path.join(file_path, '*.txt')):
    with open(filename, 'r', encoding="ISO-8859-1") as infile:
        emails.append(infile.read())
        labels.append(1)

file_path = 'enron1/ham/'
for filename in glob.glob(os.path.join(file_path, '*.txt')):
    with open(filename, 'r', encoding="ISO-8859-1") as infile:
        emails.append(infile.read())
        labels.append(0)

print(len(labels))

print(len(emails))




def is_letter_only(word):
    return word.isalpha()

from nltk.corpus import names
all_names = set(names.words())

from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

def clean_text(docs):
    docs_cleaned = []
    for doc in docs:
        doc = doc.lower()
        doc_cleaned = ' '.join(lemmatizer.lemmatize(word) for word in doc.split() if is_letter_only(word) and word not in all_names)
        docs_cleaned.append(doc_cleaned)
    return docs_cleaned

emails_cleaned = clean_text(emails)

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(stop_words="english", max_features=1000, max_df=0.5, min_df=2)

docs_cv = cv.fit_transform(emails_cleaned)
print(docs_cv[0])

terms = cv.get_feature_names()
print(terms[932])
print(terms[968])
print(terms[715])



def get_label_index(labels):
    from collections import defaultdict
    label_index = defaultdict(list)
    for index, label in enumerate(labels):
        label_index[label].append(index)
    return label_index


def get_prior(label_index):
    """
    Compute prior based on training samples
    @param label_index: grouped sample indices by class
    @return: dictionary, with class label as key, corresponding prior as the value
    """
    prior = {label: len(index) for label, index in label_index.items()}
    total_count = sum(prior.values())
    for label in prior:
        prior[label] /= float(total_count)
    return prior


label_index = get_label_index(labels)
prior = get_prior(label_index)
print('Prior:', prior)


def get_likelihood(term_matrix, label_index, smoothing=0):
    """
    Compute likelihood based on training samples
    @param term_matrix: sparse matrix of the term frequency features
    @param label_index: grouped sample indices by class
    @param smoothing: integer, additive Laplace smoothing parameter
    @return: dictionary, with class as key, corresponding conditional probability P(feature|class) vector as value
    """
    likelihood = {}
    for label, index in label_index.items():
        likelihood[label] = term_matrix[index, :].sum(axis=0) + smoothing
        likelihood[label] = np.asarray(likelihood[label])[0]
        total_count = likelihood[label].sum()
        likelihood[label] = likelihood[label] / float(total_count)
    return likelihood

smoothing = 1
likelihood = get_likelihood(docs_cv, label_index, smoothing)

print(len(likelihood[0]))

print(likelihood[0][:5])
print(likelihood[1][:5])



def get_posterior(term_matrix, prior, likelihood):
    """
    Compute posterior of testing samples, based on prior and likelihood
    @param term_matrix: sparse matrix of the term frequency features
    @param prior: dictionary, with class label as key, corresponding prior as the value
    @param likelihood: dictionary, with class label as key, corresponding conditional probability vector as value
    @return: dictionary, with class label as key, corresponding posterior as value
    """
    num_docs = term_matrix.shape[0]
    posteriors = []
    for i in range(num_docs):
        # posterior is proportional to prior * likelihood
        # = exp(log(prior * likelihood))
        # = exp(log(prior) + log(likelihood))
        posterior = {key: np.log(prior_label) for key, prior_label in prior.items()}
        for label, likelihood_label in likelihood.items():
            term_document_vector = term_matrix.getrow(i)
            counts = term_document_vector.data
            indices = term_document_vector.indices
            for count, index in zip(counts, indices):
                posterior[label] += np.log(likelihood_label[index]) * count
        # exp(-1000):exp(-999) will cause zero division error,
        # however it equates to exp(0):exp(1)
        min_log_posterior = min(posterior.values())
        for label in posterior:
            try:
                posterior[label] = np.exp(posterior[label] - min_log_posterior)
            except:
                posterior[label] = float('inf')
        # normalize so that all sums up to 1
        sum_posterior = sum(posterior.values())
        for label in posterior:
            if posterior[label] == float('inf'):
                posterior[label] = 1.0
            else:
                posterior[label] /= sum_posterior
        posteriors.append(posterior.copy())
    return posteriors



emails_test = [
    '''Subject: flat screens
    hello ,
    please call or contact regarding the other flat screens requested .
    trisha tlapek - eb 3132 b
    michael sergeev - eb 3132 a
    also the sun blocker that was taken away from eb 3131 a .
    trisha should two monitors also michael .
    thanks
    kevin moore''',
    '''Subject: let ' s stop the mlm insanity !
    still believe you can earn $ 100 , 000 fast in mlm ? get real !
    get emm , a brand new system that replaces mlm with something that works !
    start earning 1 , 000 ' s now ! up to $ 10 , 000 per week doing simple online tasks .
    free info - breakfree @ luxmail . com - type " send emm info " in the subject box .
    this message is sent in compliance of the proposed bill section 301 . per section 301 , paragraph ( a ) ( 2 ) ( c ) of s . 1618 . further transmission to you by the sender of this e - mail may be stopped at no cost to you by sending a reply to : " email address " with the word remove in the subject line .
    ''',
]

emails_cleaned_test = clean_text(emails_test)
term_docs_test = cv.transform(emails_cleaned_test)


posterior = get_posterior(term_docs_test, prior, likelihood)
print(posterior)



from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(emails_cleaned, labels, test_size=0.33, random_state=42)

print(len(X_train), len(Y_train))
len(X_test), len(Y_test)

term_docs_train = cv.fit_transform(X_train)

label_index = get_label_index(Y_train)
prior = get_prior(label_index)
likelihood = get_likelihood(term_docs_train, label_index, smoothing)

term_docs_test = cv.transform(X_test)


posterior = get_posterior(term_docs_test, prior, likelihood)

correct = 0.0
for pred, actual in zip(posterior, Y_test):
    if actual == 1:
        if pred[1] >= 0.5:
            correct += 1
    elif pred[0] > 0.5:
        correct += 1

print('The accuracy on {0} testing samples is: {1:.1f}%'.format(len(Y_test), correct/len(Y_test)*100))




from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB(alpha=1.0, fit_prior=True)
clf.fit(term_docs_train, Y_train)
prediction_prob = clf.predict_proba(term_docs_test)

print(prediction_prob[0:10])

prediction = clf.predict(term_docs_test)

print(prediction[:10])

accuracy = clf.score(term_docs_test, Y_test)

print('The accuracy using MultinomialNB is: {0:.1f}%'.format(accuracy*100))




from sklearn.metrics import confusion_matrix
print(confusion_matrix(Y_test, prediction, labels=[0, 1]))

from sklearn.metrics import precision_score, recall_score, f1_score
precision_score(Y_test, prediction, pos_label=1)
recall_score(Y_test, prediction, pos_label=1)
f1_score(Y_test, prediction, pos_label=1)

f1_score(Y_test, prediction, pos_label=0)

from sklearn.metrics import classification_report
report = classification_report(Y_test, prediction)
print(report)




pos_prob = prediction_prob[:, 1]
thresholds = np.arange(0.0, 1.2, 0.1)
true_pos, false_pos = [0]*len(thresholds), [0]*len(thresholds)
for pred, y in zip(pos_prob, Y_test):
    for i, threshold in enumerate(thresholds):
        if pred >= threshold:
            if y == 1:
                true_pos[i] += 1
            else:
                false_pos[i] += 1
        else:
            break

true_pos_rate = [tp / 516.0 for tp in true_pos]
false_pos_rate = [fp / 1191.0 for fp in false_pos]


import matplotlib.pyplot as plt
plt.figure()
lw = 2
plt.plot(false_pos_rate, true_pos_rate, color='darkorange',
         lw=lw)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()




from sklearn.metrics import roc_auc_score
roc_auc_score(Y_test, pos_prob)



from sklearn.model_selection import StratifiedKFold
k = 10
k_fold = StratifiedKFold(n_splits=k, random_state=42)
cleaned_emails_np = np.array(emails_cleaned)
labels_np = np.array(labels)

max_features_option = [2000, 8000, None]
smoothing_factor_option = [0.5, 1.0, 2.0, 4.0]
fit_prior_option = [True, False]

max_features_option = [None]
smoothing_factor_option = [4.0, 10, 16, 20, 32]
fit_prior_option = [True, False]


auc_record = {}

for train_indices, test_indices in k_fold.split(emails_cleaned, labels):
    X_train, X_test = cleaned_emails_np[train_indices], cleaned_emails_np[test_indices]
    Y_train, Y_test = labels_np[train_indices], labels_np[test_indices]
    for max_features in max_features_option:
        if max_features not in auc_record:
            auc_record[max_features] = {}
        cv = CountVectorizer(stop_words="english", max_features=max_features, max_df=0.5, min_df=2)
        term_docs_train = cv.fit_transform(X_train)
        term_docs_test = cv.transform(X_test)
        for alpha in smoothing_factor_option:
            if alpha not in auc_record[max_features]:
                auc_record[max_features][alpha] = {}
            for fit_prior in fit_prior_option:
                clf = MultinomialNB(alpha=alpha, fit_prior=fit_prior)
                clf.fit(term_docs_train, Y_train)
                prediction_prob = clf.predict_proba(term_docs_test)
                pos_prob = prediction_prob[:, 1]
                auc = roc_auc_score(Y_test, pos_prob)
                auc_record[max_features][alpha][fit_prior] = auc + auc_record[max_features][alpha].get(fit_prior, 0.0)



print('max features  smoothing  fit prior  auc')
for max_features, max_feature_record in auc_record.items():
    for smoothing, smoothing_record in max_feature_record.items():
        for fit_prior, auc in smoothing_record.items():
            print('       {0}      {1}      {2}    {3:.5f}'.format(max_features, smoothing, fit_prior, auc/k))

