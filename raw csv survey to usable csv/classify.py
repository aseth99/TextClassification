# https://towardsdatascience.com/machine-learning-nlp-text-classification-using-scikit-learn-python-and-nltk-c52b92a7c73a
import numpy as np  
import re  
import nltk  
from sklearn.datasets import load_files  
# nltk.download('stopwords')  
import pickle  
from nltk.corpus import stopwords  

from sklearn.datasets import fetch_20newsgroups
news = fetch_20newsgroups(subset='all') #errything
twenty_train = fetch_20newsgroups(subset='train', shuffle=True) #training data, the ones with labels
# print(twenty_train.target_names) #prints all the categories
# print("\n".join(twenty_train.data[0].split("\n"))) #prints first line of the first data file


print(len(news.data))
# 18846
 
print(len(news.target_names))
# 20
 
print(news.target_names)
# ['alt.atheism', 'comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware', 'comp.windows.x', 'misc.forsale', 'rec.autos', 'rec.motorcycles', 'rec.sport.baseball', 'rec.sport.hockey', 'sci.crypt', 'sci.electronics', 'sci.med', 'sci.space', 'soc.religion.christian', 'talk.politics.guns', 'talk.politics.mideast', 'talk.politics.misc', 'talk.religion.misc']
 
# for text, num_label in zip(news.data[:10], news.target[:10]):
#     print('[%s]:\t\t "%s ..."' % (news.target_names[num_label], text[:300].split('\n')[0]))

from sklearn.feature_extraction.text import CountVectorizer #scikit learn has a countvectorizer component that creates feature vectors for us
# count_vect = CountVectorizer()
# X_train_counts = count_vect.fit_transform(twenty_train.data)
# print(X_train_counts.shape)


from sklearn.feature_extraction.text import TfidfTransformer
# tfidf_transformer = TfidfTransformer()
# X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
# print(X_train_tfidf.shape)


from sklearn.naive_bayes import MultinomialNB
# clf = MultinomialNB().fit(X_train_tfidf, twenty_train.target)


from sklearn.pipeline import Pipeline
text_clf = Pipeline([('vect', CountVectorizer(stop_words = 'english')),('tfidf', TfidfTransformer()), ('clf', MultinomialNB())])
text_clf = text_clf.fit(twenty_train.data, twenty_train.target)


import numpy as np
twenty_test = fetch_20newsgroups(subset='test', shuffle=True)
predicted = text_clf.predict(twenty_test.data)
print(np.mean(predicted == twenty_test.target))

# #custom data from a csv file... 
# import pandas as pd
# data = pd.read_csv(‘your.csv’) #text in column 1, classifier in column 2.
# import numpy as np
# numpy_array = data.as_matrix()
# X = numpy_array[:,0]
# Y = numpy_array[:,1]

# from sklearn.model_selection import train_test_split
# X_train, X_test, Y_train, Y_test = train_test_split(
#  X, Y, test_size=0.4, random_state=42)

# from sklearn.feature_extraction.text import CountVectorizer

# from sklearn.feature_extraction.text import TfidfTransformer

# from sklearn.naive_bayes import MultinomialNB

# from sklearn.pipeline import Pipeline
# text_clf = Pipeline([(‘vect’, CountVectorizer(stop_words=’english’)),
#  (‘tfidf’, TfidfTransformer()),
#  (‘clf’, MultinomialNB()),
# ])

# text_clf = text_clf.fit(X_train,Y_train)

# predicted = text_clf.predict(X_test)
# np.mean(predicted == Y_test)
