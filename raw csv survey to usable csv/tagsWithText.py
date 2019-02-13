#custom data from a csv file... 
from textCleaning import prepare_text_for_lda
import pandas as pd
import numpy as np
import math

# Timestamp,What is the Article ID?,Potential Topics (from contify datasheet),Regions,Product Category
data = pd.read_csv('survey.csv') #text in column 1, classifier in column 2.

numpy_array = data.as_matrix()
articleID = numpy_array[:,1]
topics = numpy_array[:,2]
regions = numpy_array[:,3]
productCategory = numpy_array[:,4]


fp = open('allArticles.txt')

articleids = []
realarticleids = []
articletexts = []
topicsArr = []
regionsArr = []
categoryArr = []

for x in numpy_array:

	if math.isnan(x[1]):
		continue
	articleids.append(int(x[1]))


for i,line in enumerate(fp):
	if i+1 in articleids:
		realarticleids.append(i+1)
		tokens = prepare_text_for_lda(line)
		articletexts.append(tokens)

done = []
for x in numpy_array:
	if math.isnan(x[1]):
		continue
	if (int(x[1]) in realarticleids):
		if(int(x[1]) in done):
			print(x[1])
		if int(x[1]) not in done:
			topicsArr.append(x[2])
			regionsArr.append(x[3])
			categoryArr.append(x[4])	
			done.append(int(x[1]))

# print('\n')
# done = []
# articleIDRegions = []
# for x in numpy_array:
# 	if math.isnan(x[3]):
# 		continue
# 	if (int(x[1]) in realarticleids):
# 		if(int(x[1]) in done):
# 			print(x[1])
# 		if int(x[1]) not in done:
# 			# topicsArr.append(x[2])
# 			articleIDRegions.append(int(x[1]))
# 			regionsArr.append(x[3])
# 			# categoryArr.append(x[4])	
# 			done.append(int(x[1]))

# print('\n')
# done = []
# articleIDCategories = []
# for x in numpy_array:
# 	if math.isnan(x[3]):
# 		continue
# 	if (int(x[1]) in realarticleids):
# 		if(int(x[1]) in done):
# 			print(x[1])
# 		if int(x[1]) not in done:
# 			# topicsArr.append(x[2])
# 			articleIDCategories.append(int(x[1]))
# 			# regionsArr.append(x[3])
# 			categoryArr.append(x[4])	
# 			done.append(int(x[1]))


# 1119.0
# 1333.0
# 459.0
# 459.0
# 658.0
# 473.0
# 117.0
# 114.0
# 459.0
# 498.0
# 707.0
# 274.0

a = np.array(realarticleids)
b = np.array(articletexts)
c = np.array(topicsArr)
d = np.array(regionsArr)
e = np.array(categoryArr)
print(len(realarticleids))
print(len(articletexts))
print(len(topicsArr))


df = pd.DataFrame({"Article IDs" : a, "Text" : b, "Topics": c, "Regions": d, "Categories": e})
df.to_csv("textsWithTags.csv", index=False)

