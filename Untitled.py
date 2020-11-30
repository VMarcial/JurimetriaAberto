#!/usr/bin/env python
# coding: utf-8

# In[1]:


from tika import parser
import re
import spacy
import pandas as pd
import pdb
import csv
import numpy as np


# In[3]:


global x
x = pd.read_csv("acordaos_teste.csv", encoding="ISO-8859-1")

train_data = [x.julgado, x.resultado]
pd.

# In[4]:


def getRelatorio(sent):
    tempo = sent.lower()
    fim = re.search('é o relatório', tempo)
    ementa = re.search('provido', tempo)
    return sent[ementa.start()+7:fim.start()+14]


# In[5]:


def teste(n):
    global x
    print(getRelatorio(x.julgado[n]))
teste(2)

relatorios = pd.DataFrame()


n = 43
i=0
while i < n:
    if getRelatorio(x.julgado[i]) == NoneType:
        relatorios[i] = NA
    else:
        relatorios[i] = getRelatorio(x.julgado[i])
        i = i+1

relatorios[0] = getRelatorio(x.julgado[0])

print(relatorios[1])


# In[6]:


npl = spacy.load("pt_core_news_sm")


# In[13]:

text1 = pd.DataFrame()
    
for i in range(43):
    text1[i] = npl(getRelatorio(x.julgado[i]))


# In[18]:


for token in text1:
    print(token.text, token.pos_, token.dep_)


# In[17]:





# In[19]:


for token in text1:
    print(token.text)


# In[20]:


from spacy import displacy
from pathlib import Path


# In[41]:


sentence_spans = list(text1.sents)
svg = displacy.render(sentence_spans[17], style="ent", jupyter = False)
output_path = Path("imagem_teste.svg")
output_path.open("w", encoding="utf-8").write(svg)


# In[ ]:

from sklearn.feature_extraction.text import CountVectorizer
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(x.julgado[range(20)])
print(X_train_counts)


from sklearn.feature_extraction.text import TfidfTransformer
tf_transformer = TfidfTransformer(use_idf=False).fit(X_train_counts)
X_train_tf = tf_transformer.transform(X_train_counts)
X_train_tf.shape

tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
X_train_tfidf.shape


from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB().fit(X_train_tfidf, x.resultado[range(20)])

docs_new = x.julgado[range(21,43)]
X_new_counts = count_vect.transform(docs_new)
X_new_tfidf = tfidf_transformer.transform(X_new_counts)

predicted = clf.predict(X_new_tfidf)

np.mean(predicted == twenty_test.target)




# %%
