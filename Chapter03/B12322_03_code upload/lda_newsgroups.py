'''
Source codes for Python Machine Learning By Example 2nd Edition (Packt Publishing)
Chapter 3: Mining the 20 Newsgroups Dataset with Clustering and Topic Modeling Algorithms
Author: Yuxi (Hayden) Liu
'''

from sklearn.datasets import fetch_20newsgroups

categories = [
    'alt.atheism',
    'talk.religion.misc',
    'comp.graphics',
    'sci.space',
]


groups = fetch_20newsgroups(subset='all', categories=categories)



def is_letter_only(word):
    for char in word:
        if not char.isalpha():
            return False
    return True



from nltk.corpus import names
all_names = set(names.words())



from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

data_cleaned = []

for doc in groups.data:
    doc = doc.lower()
    doc_cleaned = ' '.join(lemmatizer.lemmatize(word) for word in doc.split() if is_letter_only(word) and word not in all_names)
    data_cleaned.append(doc_cleaned)



from sklearn.feature_extraction.text import CountVectorizer
count_vector = CountVectorizer(stop_words="english", max_features=None, max_df=0.5, min_df=2)


data = count_vector.fit_transform(data_cleaned)


from sklearn.decomposition import LatentDirichletAllocation

t = 20
lda = LatentDirichletAllocation(n_components=t, learning_method='batch',random_state=42)

lda.fit(data)

print(lda.components_)

terms = count_vector.get_feature_names()


for topic_idx, topic in enumerate(lda.components_):
        print("Topic {}:" .format(topic_idx))
        print(" ".join([terms[i] for i in topic.argsort()[-10:]]))


