import tensorflow as tf
import tensorflow_hub as hub
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import spacy
import pdb
import csv
from sklearn.naive_bayes import MultinomialNB
import re

train_df = pd.read_csv("acordaos_treinamento.csv", encoding="ISO-8859-1")
test_df = pd.read_csv("acordaos_teste.csv", encoding="ISO-8859-1")

def getRelatorio(sent):
    tempo = sent.lower()
    fim = re.search('é o relatório', tempo)
    # TODO aprimorar ementa
    ementa = re.search('provido', tempo)
    if fim == None or ementa == None:
        return None
    return sent[ementa.start()+7:fim.start()+14]

def relatorios(n, cases):
    final = []
    i = 0
    while i < n:
        if getRelatorio(cases.julgado[i]) != None:
            final.append(getRelatorio(cases.julgado[i]))
        i += 1
    return final

npl = spacy.load("pt_core_news_sm")
relat = relatorios(1000, train_df)
relat[6]

train_df.resultado[range(50)]

opcoes = ['provido', 'improvido', 'parcial', 'não conhecido', 'desconhecido']

i=0
relat_npl = []
while i < 4000:
    x = npl(relat[i])
    relat_npl.append(x)
    i = i+1

import os
import numpy as np

import tensorflow as tf
import tensorflow_hub as hub



hub_layer = hub.KerasLayer(train_df.julgado, input_shape=[], dtype=tf.string, trainable=True)
hub_layer(train_examples_batch[:3])









#####

from sklearn.feature_extraction.text import CountVectorizer
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(train_df.julgado)
print(X_train_counts)


from sklearn.feature_extraction.text import TfidfTransformer
tf_transformer = TfidfTransformer(use_idf=False).fit(X_train_counts)
X_train_tf = tf_transformer.transform(X_train_counts)
X_train_tf.shape

tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
X_train_tfidf.shape


from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB().fit(X_train_tfidf, train_df.resultado)

docs_new = test_df.resultado
X_new_counts = count_vect.transform(docs_new)
X_new_tfidf = tfidf_transformer.transform(X_new_counts)

predicted = clf.predict(X_new_tfidf)

print(predicted)

np.mean(predicted == docs_new)






