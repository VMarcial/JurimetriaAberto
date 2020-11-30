# Usando python 3.8.6
# Tem que usar numpy 1.19.3 pra não dar problema com windows runtime


import csv
#from tika import parser Isso é o troço pra ler PDF,
#  acho que não vai mais precisar 
import re
import spacy
import pandas as pd
import pdb
import numpy as np
import csv
import string
from sklearn import metrics
from sklearn.base import TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split
from spacy.lang.pt import Portuguese
from spacy.lang.pt.stop_words import STOP_WORDS

def tfDecisions(julgado):
    # Transforma os diferentes tipo de nome das decisões nos targets pros modelos
    # 1 Provido
    # 2 Improvido ou não conhecido
    # 3 Parcial
    # 4 Desconhecido/ Não conhecido
    # 5 Outros casos jurídicos
    prov = ["provido"]
    improv = ["improvido", "não conhecido"]
    parcial = ["parcial"]
    desconhecido = ["não conhecido", "desconhecido"]
    fim = ["prejudicado/extinto", "duvida", "anulado", "desistência"]
    lista = []
    for i in julgado:
        k = str(i)
        k = k.lower()
        if k in prov:
            lista.append(1)
        elif k in improv:
            lista.append(2)
        elif k in parcial:
            lista.append(3)
        elif k in desconhecido:
            lista.append(4)
        elif k in fim:
            lista.append(5)
        else:
            print("Outro tipo de caso jurídico: ", k, "\nO código precisará de modificações para acomodar isto.")
            lista.append(None)
    return lista


def main():
    # É melhor rodar sempre no modo iterativo ("python -i nomedoarquivo")

    print("Carregando NLP")
    nlp = spacy.load("pt_core_news_sm")
    print("Carregando CSV")
    csvBase = pd.read_csv("relatorios_full.csv")
    
    #TODO Se tudo der certo depois colocados o spacy por ser melhor que o sklearn pra npl
    #TODO se funcionar o spacy, separar por pessoa que fala no relatório


    # O formato do data_relat é pra ser um  df com relatorio e julgado
    #data_relat = pd.read_csv("relatorios_teste.csv", encoding="ISO-8859-1")
    data_relat = csvBase
    decisoes = tfDecisions(data_relat.resultado)


    # Utilizando métodos do sklearn para nlp
    countVect = CountVectorizer()
    xFullCount = countVect.fit_transform(data_relat.julgado.astype('U').values)
    tf_transformer = TfidfTransformer(use_idf=False).fit(xFullCount)
    xFull = tf_transformer.transform(xFullCount)
    X_train, X_test, Y_train, Y_test = train_test_split(xFull, decisoes, stratify=decisoes,random_state=1,train_size = 0.9) # split no texto vetorizado em bags of words 
  
    # Com Spacy

    data_relat = data_relat.dropna()
    
    relat_nlp = data_relat["julgado"].apply(nlp)

    # Remoção de stopwords e lemmatization
    #print("Limpando o NLP")
    #relat_nlp1 = [] # lemmas sem stopwords
    #relat_nlp2 = [] # dependency parsing
    #for i in relat_nlp:
    #    temp = []
    #    for token in i:
    #        if token.is_stop == False:
    #            temp.append(token.lemma_.lower())
    #    relat_nlp1.append(temp)
    #
    #
    #    temp2 = []
    #    for chunk in i.noun_chunks:
    #        temp2.append(chunk) # Se a gente conseguisse o nome das partes talvez isso funcionasse melhor
    #    relat_nlp2.append(temp2)
    #
    #i = 0
    #while i < len(relat_nlp1):
    #    relat_nlp1[i] = str(relat_nlp1[i])
    #    i += 1

    
    print("Vetorizando o NLP")

    #countVect = CountVectorizer()
    #xFullCount = countVect.fit_transform(relat_nlp)# df dos relatorios
    #tf_transformer = TfidfTransformer(use_idf=False).fit(xFullCount)
    #xFull = tf_transformer.transform(xFullCount)

    print("Separando as amostras")
    #X_train, X_test, Y_train, Y_test = train_test_split(xFull, decisoes, stratify=decisoes,random_state=1,train_size = 0.9) # split no texto vetorizado em bags of words 

    classifier = MLPClassifier(solver='sgd', alpha=1e-5, tol = 1e-10,hidden_layer_sizes=(200,200,200,200,200,200,200,200,200,200), random_state=543, max_iter=100000, verbose = True)
    bow_vector = CountVectorizer(tokenizer = spacy_tokenizer, ngram_range=(1,1))

    # Create pipeline using Bag of Words
    pipe = Pipeline([("cleaner", predictors()),
                     ('vectorizer', bow_vector),
                     ('classifier', classifier)])
    
    pipe.fit(X_train, Y_train)

    predicted = pipe.predict(X_test)

    print("Regression Accuracy:",metrics.accuracy_score(Y_test, predicted))
    print("Regression Precision:",metrics.precision_score(Y_test, predicted))
    print("Regression Recall:",metrics.recall_score(Y_test, predicted))
    
    # Transforma as decisões de texto pra números que podem ser usados de target

    # Treinamento com Naive bayes
    # NBclf = MultinomialNB().fit(xTrain, decisoes)

    # Treinamento com neural network
    #print("Treinando a Neural Network")
    #NNclf = MLPClassifier(solver='sgd', alpha=1e-5, tol = 1e-10,hidden_layer_sizes=(200,200,200,200,200,200,200,200,200,200), random_state=543, max_iter=100000, verbose = True)
    #NNclf.fit(X_train, Y_train) # X = relatorio, Y= resultado

    #  xTestCount = countVect.fit_transform(basedeteste.julgado.astype('U').values)# df dos relatorios
    #  test_transformer = TfidfTransformer(use_idf=False).fit(xTestCount)
    #  xTest = test_transformer.transform(xTestCount)
    #  decisoes_test = tfDecisions(basedeteste.resultado)

    # Depois disso pode rodar NNclf.predict(basedeteste) e NBclf.predict(basedeteste)
    #print("Fazendo as predições")
    #pred = NNclf.predict(X_test)

    #np.mean(pred == Y_test)
    #TODO metricas da predição

def spacy_tokenizer(sentence):
    parser = Portuguese()
    punctuations = string.punctuation
    stop_words = spacy.lang.pt.stop_words.STOP_WORDS
    # Creating our token object, which is used to create documents with linguistic annotations.
    mytokens = parser(sentence)

    # Lemmatizing each token and converting each token into lowercase
    mytokens = [ word.lemma_.lower().strip() if word.lemma_ != "-PRON-" else word.lower_ for word in mytokens ]

    # Removing stop words
    mytokens = [ word for word in mytokens if word not in stop_words and word not in punctuations ]

    # return preprocessed list of tokens
    return mytokens


# Custom transformer using spaCy
class predictors(TransformerMixin):
    def transform(self, X, **transform_params):
        # Cleaning Text
        temp = []
        for text in X:
            temp.append(clean_text(text))
        return temp

    def fit(self, X, y=None, **fit_params):
        return self

    def get_params(self, deep=True):
        return {}

# Basic function to clean the text
def clean_text(text):
    # Removing spaces and converting text into lowercase
    print(text)
    return text

if __name__ == "__main__":
    main()


