import numpy as np
import nltk
from sklearn.naive_bayes import MultinomialNB
from sklearn import svm
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import accuracy_score, recall_score, precision_score
from nltk.parse.stanford import StanfordDependencyParser
from nltk import classify

ambiguous_word = []
true_polarity = []
reviews = []


def precision_recall_f1(predicted, true, tag):
    labels = list(zip(predicted, true))
    target = [x for j, x in enumerate(labels) if labels[j][0] == tag]
    true = [x for j, x in enumerate(labels) if labels[j][1] == tag]
    precision_list = list(zip(*target))
    recall_list = list(zip(*true))

    precision = precision_score(precision_list[1], precision_list[0], pos_label=tag)
    recall = recall_score(recall_list[1], recall_list[0], pos_label=tag)
    print(tag)
    print("Precision: ", precision)
    print("Recall: ", recall)
    print("F1_Measure: ", (2 * precision * recall)/(precision + recall))


def baseline():
    n_gram_vectorizer = CountVectorizer(ngram_range=(1, 2), token_pattern=r'\b\w+\b', stop_words='english')
    n_gram_tf_idf = n_gram_vectorizer.fit_transform(reviews).toarray()
    n_gram_tf_idf[n_gram_tf_idf > 0] = 1

    predicted_polarity = []
    for i in range(0, len(train_file)-1):
        print(i)
        tf_idf = [x for j, x in enumerate(n_gram_tf_idf) if j != i]
        polarity = [x for j, x in enumerate(true_polarity) if j != i]

        # mnb = MultinomialNB(alpha=1)
        # mnb_model = mnb.fit(tf_idf, polarity)
        # predicted_polarity.append(mnb_model.predict(n_gram_tf_idf[i].reshape(1, -1))[0])

        clf = svm.SVC(kernel='linear', C=1)
        clf.fit(tf_idf, polarity)
        predicted_polarity.append(clf.predict(n_gram_tf_idf[i].reshape(1, -1)))

    print("Accuracy: ", accuracy_score(predicted_polarity, true_polarity))
    precision_recall_f1(predicted_polarity, true_polarity, "POS")
    precision_recall_f1(predicted_polarity, true_polarity, "NEG")


def negative_positive():
    predicted_polarity = []
    for i in range(0, len(reviews)):
        negative = {}
        positive = {}
        for r in range(0, len(reviews)):
            if i != r:
                print(r)
                result = dependency_parser.raw_parse(reviews[r])
                dep = result.__next__()
                parsed = list(dep.triples())
                for i in range(0, len(parsed)):
                    if parsed[i][1] == 'nsubj' and parsed[i][0][1] == 'JJ':
                        if true_polarity[r] == 'NEG':
                            if parsed[i][2][0] in negative:
                                negative[parsed[i][2][0]].append(parsed[i][0][0])
                            else:
                                negative[parsed[i][2][0]] = set()
                                negative[parsed[i][2][0]].add(parsed[i][0][0])
                        else:
                            if parsed[i][2][0] in positive:
                                positive[parsed[i][2][0]].append(parsed[i][0][0])
                            else:
                                positive[parsed[i][2][0]] = set()
                                positive[parsed[i][2][0]].add(parsed[i][0][0])
        result = dependency_parser.raw_parse(reviews[i])
        dep = result.__next__()
        parsed = list(dep.triples())
        for ii in range(0, len(parsed)):
            if parsed[ii][1] == 'nsubj' and parsed[ii][0][1] == 'JJ':
                try:
                    adjectives = negative[parsed[ii][2][0]]
                    if parsed[ii][0][2] in adjectives:
                        predicted_polarity.append("NEG")
                except:
                    predicted_polarity.append("POS")

    print("Accuracy: ", accuracy_score(predicted_polarity, true_polarity))
    precision_recall_f1(predicted_polarity, true_polarity, "POS")
    precision_recall_f1(predicted_polarity, true_polarity, "NEG")


def max_ent():
    train = []
    for r in range(0, len(reviews)):
        features = dict()
        print(r)
        result = dependency_parser.raw_parse(reviews[r])
        dep = result.__next__()
        parsed = list(dep.triples())
        for p in range(0, len(parsed)):
            if parsed[p][1] == 'nsubj' and parsed[p][0][1] == 'JJ':
                new_entry = {"word": parsed[p][2][0].lower(), "adj": parsed[p][0][0].lower()}
                features.update(new_entry)
        train.append((features, true_polarity[r]))

    predicted_polarity = []
    for t in range(0, len(train)):
        new_train = [x for j, x in enumerate(train) if j != t]
        classifier = nltk.MaxentClassifier.train(new_train, max_iter=10)
        predicted_polarity.append(classifier.classify(train[t][0]))

    print("Accuracy: ", accuracy_score(predicted_polarity, true_polarity))
    precision_recall_f1(predicted_polarity, true_polarity, "POS")
    precision_recall_f1(predicted_polarity, true_polarity, "NEG")


if __name__ == '__main__':

    with open('./dataset/dataset.txt', 'r') as fileTrain:
        train_file = fileTrain.read().split('\n')
        fileTrain.close()
    for i in range(0, len(train_file)-1):
        current = train_file[i].split('\t')
        ambiguous_word.append(current[2].split('<>')[2])
        true_polarity.append(current[3])
        reviews.append(current[4])

    path_to_jar = './stanford-parser-full-2018-02-27/stanford-parser.jar'
    path_to_models_jar = './stanford-parser-full-2018-02-27/stanford-parser-3.9.1-models.jar'
    dependency_parser = StanfordDependencyParser(path_to_jar=path_to_jar, path_to_models_jar=path_to_models_jar)
    baseline()
    # negative_positive()
    # max_ent()
