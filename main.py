from nltk import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
import os
import nltk
import string
string.punctuation
import inflect
p = inflect.engine()


# defining the function to convert lower case
def convert_lower_case(data):
    new_data = str(np.char.lower(str(data)))
    return new_data


# remove stopwords function
def remove_stopwords(text):
    stop_words = set(stopwords.words("english"))
    word_tokens = word_tokenize(text)
    filtered_text = str([word for word in word_tokens if word not in stop_words])
    return filtered_text



# defining the function to remove punctuation
def remove_punctuation(inp):
    # punctuationfree = "".join([i for i in data if i not in string.punctuation])
    # return punctuationfree
    tokenizer = RegexpTokenizer("\w+")  ## \w+ matches alphanumeric characters a-z,A-Z,0-9 and _
    tokens = tokenizer.tokenize(inp)
    new_word = " ".join(tokens)
    return new_word


#defining the function to remove numbers
def remove_number(data):
    result = ''.join([i for i in data if not i.isdigit()])
    return result


# function for lemmatization
def lemmatize_word(text):
    lemmatizer = WordNetLemmatizer()
    word_tokens = word_tokenize(text)
    lemmas = str([lemmatizer.lemmatize(word, pos='v') for word in word_tokens])
    return lemmas



# function to preprocess the raw documents
def preprocess(data):
    data = convert_lower_case(data)
    data = remove_number(data)
    data = remove_punctuation(data)
    data = remove_stopwords(data)
    data = remove_punctuation(data)
    data = lemmatize_word(data)
    data = remove_punctuation(data)
    return data


# Reading files
data = list()
for folder in os.listdir('./Data'):
    for file in os.listdir(os.path.join('./Data', folder)):
        with open(os.path.join('./Data', folder, file), encoding="utf-8") as text:
            words = text.read()
            data.append([words, folder])

df = pd.DataFrame(data, columns=['Text', 'Category'])


#Technology IDF calculation
Tech = df[df.Category.str.startswith('T')]
Tech['clean_tech'] = Tech['Text'].apply(lambda x: preprocess(x))

cv = CountVectorizer()
word_count_vector = cv.fit_transform(Tech['clean_tech'])
tfidf_transformer = TfidfTransformer(norm='l1',smooth_idf=True,use_idf=True,sublinear_tf=False)
tfidf_transformer.fit(word_count_vector)
# idf calculation
df_idf = pd.DataFrame(tfidf_transformer.idf_, index=cv.get_feature_names_out(),columns=["idf_weights"])
# sort ascending
Tech_Result = df_idf.sort_values(by=['idf_weights'],ascending=False)
# saving result to csv document
Tech_Result.to_csv('Tech_IDF_Result.csv')


#Business IDF calculation
Business = df[df.Category.str.startswith('B')]
Business['clean_business'] = Business['Text'].apply(lambda e: preprocess(e))

cv = CountVectorizer()
word_count_vector = cv.fit_transform(Business['clean_business'])
tfidf_transformer = TfidfTransformer(norm='l1',smooth_idf=True,use_idf=True,sublinear_tf=False)
tfidf_transformer.fit(word_count_vector)
# idf calculation
df_idf = pd.DataFrame(tfidf_transformer.idf_, index=cv.get_feature_names_out(),columns=["idf_weights"])
# sort ascending
Business_Result = df_idf.sort_values(by=['idf_weights'],ascending=False)
# saving result to csv document
Business_Result.to_csv('Business_IDF_Result.csv')


#Entertainment&Arts IDF calculation
Arts = df[df.Category.str.startswith('E')]
Arts['clean_arts'] = Arts['Text'].apply(lambda e: preprocess(e))

cv = CountVectorizer()
word_count_vector = cv.fit_transform(Arts['clean_arts'])
tfidf_transformer = TfidfTransformer(norm='l1',smooth_idf=True,use_idf=True,sublinear_tf=False)
tfidf_transformer.fit(word_count_vector)
# idf calculation
df_idf = pd.DataFrame(tfidf_transformer.idf_, index=cv.get_feature_names_out(),columns=["idf_weights"])
# sort ascending
Arts_Result = df_idf.sort_values(by=['idf_weights'],ascending=False)
# saving result to csv document
Arts_Result.to_csv('Entertainment&Arts_IDF_Result.csv')


#Health IDF calculation
Health = df[df.Category.str.startswith('H')]
Health['clean_health'] = Health['Text'].apply(lambda e: preprocess(e))

cv = CountVectorizer()
word_count_vector = cv.fit_transform(Health['clean_health'])
tfidf_transformer = TfidfTransformer(norm='l1',smooth_idf=True,use_idf=True,sublinear_tf=False)
tfidf_transformer.fit(word_count_vector)
# idf calculation
df_idf = pd.DataFrame(tfidf_transformer.idf_, index=cv.get_feature_names_out(),columns=["idf_weights"])
# sort ascending
Health_Result = df_idf.sort_values(by=['idf_weights'],ascending=False)
# saving result to csv document
Health_Result.to_csv('Health_IDF_Result.csv')


#Science&Environment IDF calculation
Science = df[df.Category.str.startswith('S')]
Science['clean_science'] = Science['Text'].apply(lambda e: preprocess(e))

cv = CountVectorizer()
word_count_vector = cv.fit_transform(Science['clean_science'])
tfidf_transformer = TfidfTransformer(norm='l1',smooth_idf=True,use_idf=True,sublinear_tf=False)
tfidf_transformer.fit(word_count_vector)
# idf calculation
df_idf = pd.DataFrame(tfidf_transformer.idf_, index=cv.get_feature_names_out(),columns=["idf_weights"])
# sort ascending
Science_Result = df_idf.sort_values(by=['idf_weights'],ascending=False)
# saving result to csv document
Science_Result.to_csv('Science&Environment_IDF_Result.csv')



# CATEGORY DISTRIBUTION GRAPH
x = df['Category'].value_counts()
print(x)
sns.barplot(x = x.index, y = x)
plt.show()


# Preprocessing texts
df['clean_text'] = df['Text'].apply(lambda x: preprocess(x))

# create Word2vec model

# convert preprocessed text to tokenized text
df['clean_text_tok'] = [nltk.word_tokenize(i) for i in df['clean_text']]

# min_count=1 means word should be present at least across all documents,
model = Word2Vec(df['clean_text_tok'], min_count=1)

# combination of word and its vector
w2v = dict(zip(model.wv.index_to_key, model.wv.vectors))


# for converting sentence to vectors/numbers from word vectors result by Word2Vec
class MeanEmbeddingVectorizer(object):
    def __init__(self, word2vec):
        self.word2vec = word2vec
        # if a text is empty we should return a vector of zeros
        # with the same dimensionality as all the other vectors
        self.dim = len(next(iter(word2vec.values())))

    def fit(self, X, y):
        return self

    def transform(self, X):
        return np.array([
            np.mean([self.word2vec[w] for w in words if w in self.word2vec]
                    or [np.zeros(self.dim)], axis=0)
            for words in X
        ])


# Splitting preprocessed data into train and test set
X_train, X_test, y_train, y_test = train_test_split(df["clean_text"], df["Category"], test_size=0.3, shuffle=True)

# Tokenizing train and test sets
X_train_tok = [nltk.word_tokenize(i) for i in X_train]
X_test_tok = [nltk.word_tokenize(i) for i in X_test]

# TF-IDF
# Convert x_train to vector since model can only run on numbers and not words- Fit and transform
tfidf_vectorizer = TfidfVectorizer(use_idf=True)

# tfidf runs on non-tokenized sentences unlike word2vec
X_train_vectors_tfidf = tfidf_vectorizer.fit_transform(X_train)
# Only transform x_test (not fit and transform)
X_test_vectors_tfidf = tfidf_vectorizer.transform(X_test)

# Word2vec
# Fit and transform
modelw = MeanEmbeddingVectorizer(w2v)
X_train_vectors_w2v = modelw.transform(X_train_tok)
X_test_vectors_w2v = modelw.transform(X_test_tok)


# CLASSIFICATION STEPS

# Implementation of Logistic Regression using TFIDF
print("Logistic Regression using TFIDF")
lr_tfidf = LogisticRegression(solver='lbfgs', max_iter=1000)
lr_tfidf.fit(X_train_vectors_tfidf, y_train)  # model
# Predict y value for test dataset
y_predict = lr_tfidf.predict(X_test_vectors_tfidf)
y_prob = lr_tfidf.predict_proba(X_test_vectors_tfidf)[:, 1]

# Calculations of Logistic Regression using TFIDF
print(classification_report(y_test, y_predict))
print('Confusion Matrix:', confusion_matrix(y_test, y_predict))
conf_mat = confusion_matrix(y_test, y_predict)
fig, ax = plt.subplots(figsize=(8,8))
sns.heatmap(conf_mat, annot=True, fmt='g')
plt.title('Plotting Confusion Matrix of Logistic Regression using TFIDF',fontsize=12);
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()

# Implementation of Logistic Regression using Word2vec
print("Logistic Regression using Word2vec")
lr_w2v = LogisticRegression(solver='lbfgs', max_iter=1000)
lr_w2v.fit(X_train_vectors_w2v, y_train)  # model
# Predict y value for test dataset
y_predict = lr_w2v.predict(X_test_vectors_w2v)
y_prob = lr_w2v.predict_proba(X_test_vectors_w2v)[:, 1]

# Calculations of Logistic Regression using Word2vec
print(classification_report(y_test, y_predict))
print('Confusion Matrix:', confusion_matrix(y_test, y_predict))
conf_mat = confusion_matrix(y_test, y_predict)
fig, ax = plt.subplots(figsize=(8,8))
sns.heatmap(conf_mat, annot=True,fmt='g')
plt.title('Plotting Confusion Matrix of Logistic Regression using Word2vec',fontsize=12);
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()

# Implementation of Naive Bayes
print("Naive Bayes using TFIDF")
nb_tfidf = MultinomialNB()
nb_tfidf.fit(X_train_vectors_tfidf, y_train)  # model

# Predict y value for test dataset
y_predict = nb_tfidf.predict(X_test_vectors_tfidf)
y_prob = nb_tfidf.predict_proba(X_test_vectors_tfidf)[:, 1]

# Calculations of Naive Bayes
print(classification_report(y_test, y_predict))
print('Confusion Matrix:', confusion_matrix(y_test, y_predict))
conf_mat = confusion_matrix(y_test, y_predict)
fig, ax = plt.subplots(figsize=(8,8))
sns.heatmap(conf_mat, annot=True,fmt='g')
plt.title('Plotting Confusion Matrix of Naive Bayes',fontsize=12);
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()


# Comparing actual category and predicted category using Logistic Regression
X_test=df['clean_text']
X_vector=tfidf_vectorizer.transform(X_test)
y_predict = lr_tfidf.predict(X_vector)
y_prob = lr_tfidf.predict_proba(X_vector)[:,1]
df['predict_prob']= y_prob
df['Predicted'] = y_predict
final = df[['Text','Category','Predicted']].reset_index(drop=True)
final.to_csv('Results.csv')
# print(final)