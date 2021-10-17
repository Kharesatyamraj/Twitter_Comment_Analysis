
"""
@author: khare
"""

## Name- SATYAM RAJ KHARE


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler 


from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer
#%pip install re
# Loading the data set
data = pd.read_csv(r"\Disaster_tweets_NB.csv",encoding = "ISO-8859-1")

# cleaning data 
import re
stop_words = []
# Load the custom built Stopwords
with open("\stop.txt","r") as sw:
    stop_words = sw.read()

stop_words = stop_words.split("\n")
   
def cleaning_text(i):
    i = re.sub("[^A-Za-z" "]+"," ",i).lower()
    i = re.sub("[0-9" "]+"," ",i)
    w = []
    for word in i.split(" "):
        if len(word)>2:
            w.append(word)
    return (" ".join(w))

#applying cleaning_text function to remove stop words and taking words greater 
# than 2 letters
data.text = data.text.apply(cleaning_text)

# removing empty rows
data = data.loc[data.text != " ",: ]


# splitting data into train and test data sets 
from sklearn.model_selection import train_test_split

twt_train, twt_test = train_test_split(data, test_size = 0.2)



# CountVectorizer
# Convert a collection of text documents to a matrix of token counts
# creating a matrix of token counts for the entire text document for
# CountVectorizer
def split_into_words(i):
    return [word for word in i.split(" ")]


# Defining the preparation of texts into word count matrix format - Bag of Words
bow = CountVectorizer(analyzer = split_into_words).fit(data.text)

# Defining BOW for all messages
all_twt_matrix = bow.transform(data.text)

# For training messages
train_matrix = bow.transform(twt_train.text)

# For testing messages
test_matrix = bow.transform(twt_test.text)

# Learning Term weighting and normalizing on entire emails
tfidf_transformer = TfidfTransformer().fit(all_twt_matrix)

# Preparing TFIDF for train emails
train_tfidf = tfidf_transformer.transform(train_matrix)
train_tfidf.shape # (row, column)

# Preparing TFIDF for test emails
test_tfidf = tfidf_transformer.transform(test_matrix)
test_tfidf.shape #  (row, column)

# Preparing a naive bayes model on training data set 

from sklearn.naive_bayes import MultinomialNB as MB

# Multinomial Naive Bayes
classifier_mb = MB()
classifier_mb.fit(train_tfidf, twt_train['target'])

# Evaluation on Test Data
test_pred_m = classifier_mb.predict(test_tfidf)

#Accuracy Test
from sklearn.metrics import accuracy_score
print(accuracy_score(test_pred_m, twt_test.target))

# Evaluation Matrix-crosstab
print(pd.crosstab(test_pred_m, twt_test.target))

# Training Data accuracy
train_pred_m = classifier_mb.predict(train_tfidf)
accuracy_train_m = np.mean(train_pred_m == twt_train.target)
print(accuracy_train_m)
# Evaluation Matrix-crosstab
print(pd.crosstab(train_pred_m, twt_train.target))

# Laplace smoothing for checking "zero probability" problem
# if alpha = 0 then no smoothing is applied and the default alpha parameter is 1

classifier_mb_lap = MB(alpha = 7) # laplace value 7
classifier_mb_lap.fit(train_tfidf, twt_train.target)

# Evaluation on Test Data after applying laplace
test_pred_lap = classifier_mb_lap.predict(test_tfidf)



from sklearn.metrics import accuracy_score
accuracy_score(test_pred_lap, twt_test.target) 

print(pd.crosstab(test_pred_lap, twt_test.target))

# Training Data accuracy
train_pred_lap = classifier_mb_lap.predict(train_tfidf)
accuracy_train_lap = np.mean(train_pred_lap == twt_train.target)
print(accuracy_train_lap)
print(pd.crosstab(train_pred_lap, twt_train.target))



