import pandas as pd
from textCleaning import prepare_text_for_lda

data = pd.read_csv('topics.csv') #text in column 1, classifier in column 2.
import numpy as np
numpy_array = data.as_matrix()
X = numpy_array[:,1]
Y = numpy_array[:,2]

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=12)

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.naive_bayes import MultinomialNB

from sklearn.pipeline import Pipeline
text_clf = Pipeline([('vect', CountVectorizer(stop_words = 'english')),('tfidf', TfidfTransformer()), ('clf', MultinomialNB())])
text_clf = text_clf.fit(X_train,Y_train)

predicted = text_clf.predict(X_test)
print(np.mean(predicted == Y_test))


sample = ["Golden Peanut and Tree Nuts, a subsidiary of Archer Daniels Midland Company (ADM) has announced that it is investing in a significant upgrade to its peanut processing facility in Alejandro Roca, Cordoba, Argentina. The upgrade plans include the addition of an in-house blancher and the construction of a cold storage warehouse.“We’re continuing to upgrade our capabilities to serve customers in Argentina and Europe,” said Carlos Urquiza, president and general manager of Golden Peanut and Tree Nuts’ operations in Argentina. “An in-house blancher and onsite cold storage will enhance both peanut quality and peanut life at our Alejandro Roca facility. Combined with the new storage facility that we announced in January, the result will be substantially enhanced capabilities for Golden Peanut and Tree Nuts in Argentina – from more efficient unloading to the very highest quality products for our customers, both locally and in the EU markets that we serve from Argentina.”The project will break ground in June 2017, and is expected to be complete in April 2018.Golden Peanut and Tree Nuts, one of the world’s premier handlers and processors of peanuts and tree nuts, has announced or completed a wide range of significant expansion and improvement projects across its global network in the last year, both in the United States and in its international operations.“We are committed to remaining the provider of choice for peanut and tree nut customers around the globe,” said Greg Mills, president, Golden Peanut and Tree Nuts. “Our customers expect the very best from Golden Peanut and Tree Nuts, and we are investing in our business to ensure we live up to those expectations.”"]

sample2 = ["Amid changing consumer tastes, Swiss food giant Nestlé has achieved its modest 2-4% growth target for three-month sales for 2017 as weak consumer demand impacts packaged foods in North America and weak prices in Western European markets affect growth. Nestlé says sales increased by 0.4% on a reported basis to CHF 21.0 billion (US$21 billion), while organic growth was solid at 2.3%, with 1.3% of real internal growth (RIG) and pricing of 1.0%.Sales were reduced by foreign exchange (-0.4%) and net divestments (-1.5%), and organic growth was 0.8% for developed markets and 4.3% for emerging markets.Despite the modest growth Nestlé CEO, Mark Schneider, remains upbeat.Organic growth of 2.3% this quarter is within our full-year guidance range. The leap year comparison and other seasonal effects made the start of this year particularly challenging,” he says.“We were encouraged by the growth in Asia and the resilience of consumer spending in Europe. Consumer demand in the Americas remained soft. Our pricing improved moderately. We confirm our 2017 guidance and have made good progress with our growth and efficiency projects to position our company for enhanced value creation.North America faced an environment of soft consumer demand, says the company, and in the US coffee creamers and frozen food maintained good momentum but confectionery and pet care declined.Brazil had a difficult quarter with subdued Easter trading and fragile economic conditions resulting in negative RIG and organic growth, Mexico's growth remained positive but decelerated, reflecting difficult comparables and weaker consumer confidence. Petcare saw good growth across Latin America.Reported sales in zone EMENA declined by 6.9% to CHF 4.0 billion (US$4 billion). Organic growth was solid at 1.7%, based on resilient RIG of 1.7% and flat pricing. Net divestments reduced reported sales by 5.9%, mainly due to the transfer of ice cream to the Froneri joint venture. Foreign exchange headwinds reduced reported sales by a further 2.7%.Pricing improved, mainly from increases taken in Nescafé throughout the zone. Pricing actions had a moderate impact on RIG. Petcare saw strong growth across the zone, particularly in Russia. Turkey and North Africa performed well, while the Middle East declined as political instability and deflation persisted.Reported sales in Nestlé Nutrition increased by 0.2% to CHF 2.6 billion (US$2.6 billion). Organic growth was 1.1%, comprised of -0.4% RIG and 1.5% pricing. Net divestments and foreign exchange reduced reported sales by 0.4% and 0.5% respectively.Price increases had a mild adverse effect on RIG in the short-term. Growth in China improved moderately, helped by increased demand for first stage products and strong momentum of illuma in the super premium segment. Our NAN Optipro roll-out continued to deliver good growth across Asia.Meanwhile, the company also said how Nespressos growth was solid as it continued to gain traction in North America. Nestlé Health Science maintained mid single-digit growth, reflecting good growth in Medical Nutrition. Nestlé Skin Health saw strong growth benefiting from low prior year comparables and several new product launches.In terms of the outlook for 2017, Nestlé expects organic growth between 2% and 4%. In order to drive future profitability, and it plans to increase restructuring costs considerably in 2017. As a result, the trading operating profit margin in constant currency is expected to be stable.Underlying earnings per share in constant currency and capital efficiency are expected to increase.Last monthFoodIngredientsFirstreported how Nestlé UK and Ireland is stepping up to the sugar challenge by stripping out 10% of sugar from its confectionery portfolio by 2018. The move will see around 7,500 tons of sugar removed across a number of well-known brands through a range of methods and initiatives.While a major breakthrough by Nestle scientists has led to bold claims of “making less sugar taste just as good” - with the potential to reduce sugar by up to 40 percent in the confectioners’ products. Nestle is patenting its findings and will begin to use the faster-dissolving sugar across a range of its confectionery products from 2018 onwards.By Gaynor Selby"]

fp = open('allArticles.txt')
textArr = []
for i,line in enumerate(fp):
	if i+1 == 15:
		print(line)
		textArr.append(line)

tokens = prepare_text_for_lda(sample[0])
print(tokens)
tokens2 = prepare_text_for_lda(sample2[0])
tokens3 = prepare_text_for_lda(textArr[0])

tokenArr = []
tokenArr.append(tokens)
tokenArr2 = []
tokenArr2.append(tokens2)
tokenArr3 = []
tokenArr3.append(tokens3)

predicted2 = text_clf.predict(tokens)
predicted3 = text_clf.predict(tokenArr2)
predicted4 = text_clf.predict(tokenArr3)

print(predicted2)

print(predicted3)

print(predicted4)