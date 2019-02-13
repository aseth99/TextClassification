# Article IDs, Text, Topics, Topics2, Topics3, Topics4, Regions, Regions1, Regions2, Regions3, Regions4, Categories, Categories2, Categories3
import pandas as pd
import numpy as np
import math

# Timestamp,What is the Article ID?,Potential Topics (from contify datasheet),Regions,Product Category
data = pd.read_csv('textsWithTags.csv') #text in column 1, classifier in column 2.

numpy_array = data.as_matrix()

articleID = numpy_array[:,0]

articleTexts = numpy_array[:,1]

topics = numpy_array[:,2]
topics2 = numpy_array[:,3]
topics3 = numpy_array[:,4]
topics4 = numpy_array[:,5]

regions = numpy_array[:,6]
regions2 = numpy_array[:,7]
regions3 = numpy_array[:,8]
regions4 = numpy_array[:,9]

categories = numpy_array[:,10]
categories2 = numpy_array[:,11]
categories3 = numpy_array[:,12]


done = []
newCategoryArr = []
newTexts = []
newArticleIdsArr = []
for x in numpy_array:
	# trueArr = np.isnan(x)
	# print(x[2])
	# print(type(x[2]))
	# if not x[2]
	if str((x[10])) != "nan":
		newArticleIdsArr.append(x[0])
		newTexts.append(x[1])
		newCategoryArr.append(x[10])
	if str((x[11])) != "nan":	
		newArticleIdsArr.append(x[0])
		newTexts.append(x[1])
		newCategoryArr.append(x[11])
	if str((x[12])) != "nan":	
		newArticleIdsArr.append(x[0])
		newTexts.append(x[1])
		newCategoryArr.append(x[12])
	

a = np.array(newArticleIdsArr)
b = np.array(newCategoryArr)
c = np.array(newTexts)
print(len(a))
print(len(b))
print(len(c))

for x, y, z in zip(a,b,c):
	print("article ID: " + str(x) + "  category: " + str(y) + "  text: " + str(z[:20]) + "   ....")


df = pd.DataFrame({"Article IDs" : a, "Category" : b, "Text": c})
df.to_csv("allCategories.csv", index=False)


