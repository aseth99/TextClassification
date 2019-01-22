

#custom data from a csv file... 
from textCleaning import prepare_text_for_lda
import pandas as pd
data = pd.read_csv('survey.csv') #text in column 1, classifier in column 2.
import numpy as np
numpy_array = data.as_matrix()
articleID = numpy_array[:,1]
topics = numpy_array[:,2]
regions = numpy_array[:,3]
productCategory = numpy_array[:,4]
# articleText = numpy_array[:,5]

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
fp = open('allArticles.txt')
# print(numpy_array)
articleids = []
realarticleids = []
articletexts = []
topicsArr = []
for x in numpy_array:
	# print(x[1])
	articleids.append(int(x[1]))
	# regionsArr.append(x[3])
	# for i, line in enumerate(fp):
	# 	if i == x[1]:
	# 		print(x[1])
	# # 		print(line)
	# 		print('HUUUH')

print(articleids)

# for y in articleids:
# 	print("got here")
for i,line in enumerate(fp):
	if i+1 in articleids:
		# print("gothere")
		print(i+1)
		print(line)
		realarticleids.append(i+1)
		tokens = prepare_text_for_lda(line)
		print(tokens)
		articletexts.append(tokens)
		# if i == y+1:
		# 	print(line)
		# 	print(y)
# print(articletexts)
# counter = 0
# text_data = []
# with open('allArticles.txt') as f:
#     for line in f:
#         # if random.random() > .99:
#         #     
#         tokens = prepare_text_for_lda(line)
#         print(tokens)
#         text_data.append(tokens)
#         counter = counter + 1
done = []
for x in numpy_array:
	if (int(x[1]) in realarticleids):
		if(int(x[1]) in done):
			print(x[1])
		if int(x[1]) not in done:
			topicsArr.append(x[2])	
			done.append(int(x[1]))

a = np.array(realarticleids)
b = np.array(articletexts)
c = np.array(topicsArr)
print(len(realarticleids))
print(len(articletexts))
print(len(topicsArr))


df = pd.DataFrame({"Article IDs" : a, "Text" : b, "Topics": c})
df.to_csv("topics.csv", index=False)

