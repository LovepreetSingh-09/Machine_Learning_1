# -*- coding: utf-8 -*-
"""
Created on Sun Jul 28 21:32:03 2019

@author: user
"""

# In the context of text analysis, the dataset is often called the corpus.
# Each data point, represented as a single text, is called a document.
# These terms come from the information retrieval (IR) and natural language processing (NLP) community, which both deal mostly in text data.
# There are four kinds of string data you might see:
# •	Categorical data
# •	Free strings that can be semantically mapped to categories
# •	Structured string data
# •	Text data

import graphviz
import mglearn
import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
import os
from IPython.display import display
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_files
print(os.getcwd())
reviews_train = load_files("aclImdb/train/")
# load_files returns a bunch, containing training texts and training labels
text_train, y_train = reviews_train.data, reviews_train.target 
print("type of text_train: {}".format(type(text_train))) 
print("length of text_train: {}".format(len(text_train))) #  25000 
print("text_train[6]:\n{}".format(text_train[6])) 
text_train = [doc.replace(b"<br />", b" ") for doc in text_train]
print(text_train[6])
print("Samples per class (training): {}".format(np.bincount(y_train)))  # [12500 12500]
reviews_test = load_files("aclImdb/test/") 
text_test, y_test = reviews_test.data, reviews_test.target 
print("Number of documents in test data: {}".format(len(text_test))) #  25000 
print("Samples per class (test): {}".format(np.bincount(y_test))) # [12500 12500]
text_test = [doc.replace(b"<br />", b" ") for doc in text_test] 
# Bag of Words :-
# Computing the bag-of-words representation for a corpus of documents consists of the following three steps:
# 1. Tokenization - Split each document into the words that appear in it (called tokens) , for example by splitting them on whitespace and punctuation.
# 2. Vocabulary building - Collect a vocabulary of all words that appear in any of the documents, and number them (say, in alphabetical order).
# 3. Encoding - For each document, count how often each of the words in the vocabulary appear in this document.
# A matrix will be built where each row represent a document and each column or feature is a word.
bards_words =["The fool doth think he is wise,", "but the wise man knows himself to be a fool"]
from sklearn.feature_extraction.text import CountVectorizer
vect=CountVectorizer()
vect.fit(bards_words)
print(len(vect.vocabulary_))
print(vect.vocabulary_)
bag_of_words=vect.transform(bards_words)
# The values are stored in sparse matrix which takes entries that are only non-zero. 
print(bag_of_words)
print(repr(bag_of_words))
print(bag_of_words.toarray())

# bag Of Words for Movie Reviews :-
vect = CountVectorizer().fit(text_train)
X_train = vect.transform(text_train) 
print("X_train:\n{}".format(repr(X_train))) 
feature_names = vect.get_feature_names()
print("Number of features: {}".format(len(feature_names))) # 74,849
print("First 20 features:\n{}".format(feature_names[:20])) 
print("Features 20010 to 20030:\n{}".format(feature_names[20010:20030])) 
print("Every 2000th feature:\n{}".format(feature_names[::2000])) 
from sklearn.model_selection import cross_val_score 
from sklearn.linear_model import LogisticRegression
scores = cross_val_score(LogisticRegression(), X_train, y_train, cv=5) 
print("Mean cross-validation accuracy: {:.2f}".format(np.mean(scores))) # 0.88
from sklearn.model_selection import GridSearchCV 
param_grid = {'C': [0.001, 0.01, 0.1, 1, 10]} 
grid = GridSearchCV(LogisticRegression(), param_grid, cv=5)
grid.fit(X_train, y_train)
print("Best cross-validation score: {:.2f}".format(grid.best_score_)) # 0.89
print("Best parameters: ", grid.best_params_) # {'C': 0.1}
X_test = vect.transform(text_test) 
print("Test score: {:.2f}".format(grid.score(X_test, y_test))) # 0.88

# df means the minimum no. of documents for each word requires to appear in the features.
vect = CountVectorizer(min_df=5).fit(text_train)
X_train = vect.transform(text_train)
print("X_train with min_df: {}".format(repr(X_train))) 
feature_names = vect.get_feature_names()
print("Number of features: {}".format(len(feature_names))) # 27271
print("First 50 features:\n{}".format(feature_names[:50])) 
print("Features 20010 to 20030:\n{}".format(feature_names[20010:20030])) 
print("Every 700th feature:\n{}".format(feature_names[::700]))
grid = GridSearchCV(LogisticRegression(), param_grid, cv=5)
grid.fit(X_train, y_train)
print("Best cross-validation score: {:.2f}".format(grid.best_score_)) # 0.89
# Stopwords :-
# Another way that we can get rid of uninformative words is by discarding words that are too frequent to be informative.
# There are two main approaches: using a languagespecific list of stopwords, or discarding words that appear too frequently.
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS 
print("Number of stop words: {}".format(len(ENGLISH_STOP_WORDS))) # 318
print("Every 10th stopword:\n{}".format(list(ENGLISH_STOP_WORDS)[::10])) 
# Specifying stop_words="english" uses the built-in list.
# We could also augment it and pass our own. 
vect = CountVectorizer(min_df=5, stop_words="english").fit(text_train)
X_train = vect.transform(text_train)
print("X_train with stop words:\n{}".format(repr(X_train)))  # 26966 features
grid = GridSearchCV(LogisticRegression(), param_grid, cv=5)
grid.fit(X_train, y_train)
print("Best cross-validation score: {:.2f}".format(grid.best_score_)) # 0.88

# TF-IDF :-  (term frequency–inverse document frequency)
# Instead of dropping features that are deemed unimportant, another approach is to rescale features by how informative we expect them to be.
# One of the most common ways to do this is using the term frequency–inverse document frequency ( tf–idf ) method.
# The intuition of this method is to give high weight to any term that appears often in a particular document, but not in many documents in the corpus.
# If a word appears often in a particular document, but not in very many documents, it is likely to be very descriptive of the content of that document.
# scikit-learn implements the tf–idf method in two classes: TfidfTransformer, which takes in the sparse matrix output produced by CountVectorizer and transforms it, and TfidfVectorizer, which takes in the text data and does both the bag-of-words feature extraction and the tf–idf transformation.
# The tf–idf score for word w in document d as :-
# tfidf (w, d) = tf * ln((N +1)/(Nw+1)) + 1 
# where N is the number of documents in the training set, Nw is the number of documents in the training set that the word w appears in .
# tf (the term frequency) is the number of times that the word w appears in the query document d ( the document you want to transform or encode).
# After making matrix of tfidf value, the L2 normalization is applied.
# which means the tfidf row of each document is divided by sum of its squared entries. Here 1st row is by sqrt(7) and 2nd row by sqrt(9). 
from sklearn.feature_extraction.text import TfidfVectorizer
tf=TfidfVectorizer()
tf.fit(bards_words)
bag=tf.transform(bards_words)
# It also gives a sparse matrix
print(bag)
print(bards_words)
print(bag.toarray())
# Extract max value of each feature
max_value=bag.max(axis=0).toarray().ravel()
print(max_value)
print(np.sort(max_value))
# Gives index or feature no. on the basis of ascending order.
sort=max_value.argsort()
print(sort)
feature_names=np.array(tf.get_feature_names())
print(feature_names)
print(feature_names[sort[:4]])
print(feature_names[sort[-4:]])
print(tf.idf_)
from sklearn.pipeline import make_pipeline 
pipe = make_pipeline(TfidfVectorizer(min_df=5) , LogisticRegression())
param_grid = {'logisticregression__C': [0.001, 0.01, 0.1, 1, 10]}
grid = GridSearchCV(pipe, param_grid, cv=5)
grid.fit(text_train, y_train)
print("Best cross-validation score: {:.2f}".format(grid.best_score_)) 
vectorizer = grid.best_estimator_.named_steps["tfidfvectorizer"]
# transform the training dataset
X_train = vectorizer.transform(text_train)
# find maximum value for each of the features over the dataset
max_value = X_train.max(axis=0).toarray().ravel()
sorted_by_tfidf = max_value.argsort()
# get feature names 
feature_names = np.array(vectorizer.get_feature_names())
print("Features with lowest tfidf:\n{}".format(feature_names[sorted_by_tfidf[:20]]))
print("Features with highest tfidf: \n{}".format(feature_names[sorted_by_tfidf[-20:]])) 
sorted_by_idf = np.argsort(vectorizer.idf_) 
print("Features with lowest idf:\n{}".format(feature_names[sorted_by_idf[:100]])) 
# Investigating Model Coefficients :-
mglearn.tools.visualize_coefficients(grid.best_estimator_.named_steps["logisticregression"].coef_,feature_names, n_top_features=40)
plt.show()
# Bag-of-Words with More Than One Word (n-Grams) :-
# Pairs of tokens are known as bigrams, triplets of tokens are known as trigrams, and more generally sequences of tokens are known as n-grams. 
print("bards_words:\n{}".format(bards_words))
cv = CountVectorizer(ngram_range=(1, 1)).fit(bards_words) 
print("Vocabulary size: {}".format(len(cv.vocabulary_))) # 13
print("Vocabulary:\n{}".format(cv.get_feature_names())) 
cv = CountVectorizer(ngram_range=(2, 2)).fit(bards_words) 
print("Vocabulary size: {}".format(len(cv.vocabulary_))) # 14
print("Vocabulary:\n{}".format(cv.get_feature_names())) 
print("Transformed data (dense):\n{}".format(cv.transform(bards_words).toarray())) 
# In principle, the number of bigrams could be the number of unigrams squared and the number of trigrams could be the number of unigrams to the power of three, leading to very large feature spaces.
# In practice, the number of higher ngrams that actually appear in the data is much smaller, because of the structure of the (English) language, though it is still large.
cv = CountVectorizer(ngram_range=(1, 3)).fit(bards_words) 
print("Vocabulary size: {}".format(len(cv.vocabulary_))) # 39 
print("Vocabulary:\n{}".format(cv.get_feature_names())) 
pipe = make_pipeline(TfidfVectorizer(min_df=5), LogisticRegression())
# running the grid search takes a long time because of the relatively large grid and the inclusion of trigrams
param_grid = {"logisticregression__C": [0.001, 0.01, 0.1, 1, 10, 100] ,"tfidfvectorizer__ngram_range": [(1, 1), (1, 2), (1, 3)]}
grid = GridSearchCV(pipe, param_grid, cv=5)
grid.fit(text_train, y_train)
print("Best cross-validation score: {:.2f}".format(grid.best_score_)) # 0.91
print("Best parameters:\n{}".format(grid.best_params_)) # {'logisticregression__C': 100, 'tfidfvectorizer__ngram_range': (1, 3)}
# extract scores from grid_search
scores = grid.cv_results_['mean_test_score'].reshape(-1, 3).T
# visualize heat map 
heatmap = mglearn.tools.heatmap(scores, xlabel="C", ylabel="ngram_range", cmap="viridis", fmt="%.3f",
    xticklabels=param_grid['logisticregression__C'] ,yticklabels=param_grid['tfidfvectorizer__ngram_range']) 
plt.colorbar(heatmap)
plt.show()
# extract feature names and coefficients
vect = grid.best_estimator_.named_steps['tfidfvectorizer'] 
feature_names = np.array(vect.get_feature_names())
coef = grid.best_estimator_.named_steps['logisticregression'].coef_ 
mglearn.tools.visualize_coefficients(coef, feature_names, n_top_features=40)

# find 3-gram features
mask = np.array([len(feature.split(" ")) for feature in feature_names]) == 3
# visualize only 3-gram features
mglearn.tools.visualize_coefficients(coef.ravel()[mask] ,
                                     feature_names[mask], n_top_features=40)

# Advanced Tokenization, Stemming, and Lemmatization :-
# Stemming :-
# The problem of nouns, verbs, related words etc. can be overcome by representing each word using its word stem, which involves identifying (or conflating) all the words that have the same word stem.
# If this is done by using a rule-based heuristic, like dropping common suffixes, it is usually referred to as stemming.
# lemmatization :-
# If instead a dictionary of known word forms is used (an explicit and human-verified system) and the role of the word in the sentence is taken into account, the process is referred to as lemmatization and the standardized form of the word is referred to as the lemma.
# Both processing methods, lemmatization and stemming, are forms of normalization that try to extract some normal form of a word. 
# To get a better understanding of normalization, let’s compare a method for stemming—the Porter stemmer, a widely used collection of heuristics (here imported from the nltk package)—to lemmatization as implemented in the spacy package.
import spacy
import nltk
import en_core_web_sm as en_nlp
# load spacy's English-language models
en_nlp = spacy.load('en_core_web_sm')
# instantiate nltk's Porter stemmer 
stemmer = nltk.stem.PorterStemmer()
print(stemmer)
# define function to compare lemmatization in spacy with stemming in nltk 
def compare_normalization(doc):     
    # tokenize document in spacy     
    doc_spacy = en_nlp(doc)
    # print lemmas found by spacy     
    print("Lemmatization:")
    print([token.lemma_ for token in doc_spacy])
    # print tokens found by Porter stemmer
    print("Stemming:")
    print([stemmer.stem(token.norm_.lower()) for token in doc_spacy])
compare_normalization(u"Our meeting today was worse than yesterday, "                        "I'm scared of meeting the clients tomorrow.") 

# Technicality: we want to use the regexp-based tokenizer
# that is used by CountVectorizer and only use the lemmatization # from spacy. To this end, we replace en_nlp.tokenizer (the spacy tokenizer)
# with the regexp-based tokenization. 
import re
# regexp used in CountVectorizer 
regexp = re.compile('(?u)\\b\\w\\w+\\b')
# load spacy language model and save old tokenizer
en_nlp = spacy.load('en') 
old_tokenizer = en_nlp.tokenizer
# replace the tokenizer with the preceding regexp
en_nlp.tokenizer = lambda string: old_tokenizer.tokens_from_list(regexp.findall(string))
# create a custom tokenizer using the spacy document processing pipeline
# (now using our own tokenizer) 
def custom_tokenizer(document):     
    doc_spacy = en_nlp(document, entity=False, parse=False)     
    return [token.lemma_ for token in doc_spacy]
# define a count vectorizer with the custom tokenizer 
lemma_vect = CountVectorizer(tokenizer=custom_tokenizer, min_df=5)
# transform text_train using CountVectorizer with lemmatization
X_train_lemma = lemma_vect.fit_transform(text_train) 
print("X_train_lemma.shape: {}".format(X_train_lemma.shape))
# standard CountVectorizer for reference 
vect = CountVectorizer(min_df=5).fit(text_train)
X_train = vect.transform(text_train) 
print("X_train.shape: {}".format(X_train.shape))
# X_train_lemma.shape:  (25000, 21596)    X_train.shape:  (25000, 27271)
# build a grid search using only 1% of the data as the training set 
from sklearn.model_selection import StratifiedShuffleSplit
param_grid = {'C': [0.001, 0.01, 0.1, 1, 10]} 
cv = StratifiedShuffleSplit(n_splits=5, test_size=0.99,train_size=0.01, random_state=0) 
grid = GridSearchCV(LogisticRegression(), param_grid, cv=cv)
# perform grid search with standard CountVectorizer
grid.fit(X_train, y_train) 
print("Best cross-validation score "
      "(standard CountVectorizer): {:.3f}".format(grid.best_score_))
# perform grid search with lemmatization 
grid.fit(X_train_lemma, y_train) 
print("Best cross-validation score "  "(lemmatization): {:.3f}".format(grid.best_score_)) 
# Best cross-validation score (standard CountVectorizer): 0.721
# Best cross-validation score (lemmatization): 0.731

# Latent Dirichlet Allocation
# Intuitively, the LDA model tries to find groups of words (the topics) that appear together frequently. 
# LDA also requires that each document can be understood as a “mixture” of a subset of the topics. 
# limit the bag-of-words model to the 10 ,000 words that are most common after removing the top 15 percent :
vect = CountVectorizer(max_features=10000, max_df=.15) 
X = vect.fit_transform(text_train)
from sklearn.decomposition import LatentDirichletAllocation 
lda = LatentDirichletAllocation(n_topics=10, learning_method="batch", max_iter=25, random_state=0)
# We build the model and transform the data in one step
# Computing transform takes some time,
# and we can save time by doing both at once 
document_topics = lda.fit_transform(X)
print("lda.components_.shape: {}".format(lda.components_.shape)) 
# lda.components_.shape: (10, 10000)
# For each topic (a row in the components_), sort the features (ascending)
# Invert rows with [:, ::-1] to make sorting descending 
sorting = np.argsort(lda.components_, axis=1)[:, ::-1]
# Get the feature names from the vectorizer 
feature_names = np.array(vect.get_feature_names()) 
# Print out the 10 topics:
mglearn.tools.print_topics(topics=range(10), feature_names=feature_names,sorting=sorting, topics_per_chunk=5, n_words=10)

lda100 = LatentDirichletAllocation(n_topics=100, learning_method="batch", max_iter=25, random_state=0) 
document_topics100 = lda100.fit_transform(X)
topics = np.array([7, 16, 24, 25, 28, 36, 37, 45, 51, 53, 54, 63, 89, 97])
sorting = np.argsort(lda100.components_, axis=1)[:, ::-1] 
feature_names = np.array(vect.get_feature_names())
mglearn.tools.print_topics(topics=topics, feature_names=feature_names,sorting=sorting, topics_per_chunk=5, n_words=20) 

# sort by weight of "music" topic 45
music = np.argsort(document_topics100[:, 45])[::-1] 
# print the five documents where the topic is most important 
for i in music[:10]:
    # show first two sentences
    print(b".".join(text_train[i].split(b".")[:2]) + b".\n")

fig, ax = plt.subplots(1, 2, figsize=(10, 10)) 
topic_names = ["{:>2} ".format(i) + " ".join(words) for i, words in enumerate(feature_names[sorting[:, :2]])] 
# two column bar chart: 
for col in [0, 1]:     
    start = col * 50     
    end = (col + 1) * 50
    ax[col].barh(np.arange(50), np.sum(document_topics100, axis=0)[start:end])
    ax[col].set_yticks(np.arange(50))
    ax[col].set_yticklabels(topic_names[start:end], ha="left", va="top")
    ax[col].invert_yaxis()     
    ax[col].set_xlim(0, 2000)    
    yax = ax[col].get_yaxis()     
    yax.set_tick_params(pad=130)
plt.tight_layout()





