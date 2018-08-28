# Sentiment-Analysis
Implementation of three Sentiment Analysis Algorithms

## Introduction
 Algorithms I implemented include:
  * baseline: First I transfered each review to a vector using tf-idf. Then I used MultinomialNB and SVM classifier from sklearn.
  * negative_positive: a simple implementation of [this](https://www.sciencedirect.com/science/article/abs/pii/S0306457316305416) paper. I divided the dataset to negative and positive reviews then I built a list of adjectives coming with each target word. For a new comment, I will find the target word and its adjective and compare with both negative and positive lists.
  * max_ent: I used target word and its adjective as a review's features. Then I used a MaxentClassifier from nltk to train a classifier and predict the polarity of new a comment with it.

## Uses
 - [Numpy](http://www.numpy.org/) version 1.14.5
 - [Sklearn](http://scikit-learn.org/stable/)
 - [Nltk](https://www.nltk.org) version 3.3
 - [Stanford Dependecy Parser](https://nlp.stanford.edu/software/lex-parser.shtml) version full 2018-02-27

## Run
 - `sentiment-analysis.py` will analyze reviews with baseline method. You can change method by uncomment the function calls.

## Output
 - Precision, Recall, F1-Measure for both positive and negative reviews
 - Accuracy