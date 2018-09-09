# Sentiment-Analysis
Implementation of three Sentiment Analysis Algorithms

## Introduction
 Algorithms I implemented include:
  * baseline: First I transferred each review to a vector using tf-idf. Then I used MultinomialNB and SVM classifier from sklearn.
  * negative_positive: a simple implementation of [this](https://www.sciencedirect.com/science/article/abs/pii/S0306457316305416) paper. I divided the dataset into negative and positive reviews. Then I built a list of adjectives, coming with each target word. For a new comment, I found the target word and related adjective as described. Then I searched both positive and negative lists to find out whether the polarity of the extracted adjective, for the target word, is positive or negative.
  * max_ent: I used target word and its adjective as a review's features. Then I used a MaxentClassifier from nltk to train a classifier and predict the polarity of a new comment with it.

## Uses
 - [Numpy](http://www.numpy.org/) version 1.14.5
 - [Sklearn](http://scikit-learn.org/stable/)
 - [Nltk](https://www.nltk.org) version 3.3
 - [Stanford Dependecy Parser](https://nlp.stanford.edu/software/lex-parser.shtml) version full 2018-02-27

## Run
 - `sentiment-analysis.py` will analyze reviews with the baseline method. You can change the method by uncommenting the function calls.

## Output
 - Precision, Recall, F1-Measure for both positive and negative reviews
 - Accuracy
