import nltk
data=nltk.corpus.brown.tagged_sents(tagset="universal")

tagged_words=[]
for sent in data:
    for tup in sent:
        tagged_words.append(tup)

tags=set()
for tup in tagged_words:
    tags.add(tup[1])

# Transition Probabilities
def calc_t2_given_t1(t2,t1,bag=tagged_words):
    lis=[]
    c_t1=0
    c_t2_and_t1=0
    for tup in tagged_words:
        lis.append(tup[1])
        if tup[1]==t1:
            c_t1+=1
    for i in range(len(lis)-1):
        if lis[i]==t1 and lis[i+1]==t2:
            c_t2_and_t1+=1
    return(c_t2_and_t1,c_t1)

taglist=[]
for tup in tagged_words:
    taglist.append(tup[1])

import numpy as np
trans=np.zeros((len(tags),len(tags)),dtype='float32')
for i,t1 in enumerate(tags):
    for j,t2 in enumerate(tags):
        trans[i,j]=calc_t2_given_t1(t2,t1)[0]/calc_t2_given_t1(t2,t1)[1]

import pandas as pd
trans_pd=pd.DataFrame(trans,columns=tags, index=tags)

trans_pd["sum"]=trans_pd.sum(axis=1)

# Probability of Word Given Tag
def calc_w_given_t(word, tag, tagset = tagged_words):
    c_w_and_t=0
    c_t=0
    for tup in tagged_words:
        if tup[0]==word and tup[1]==tag:
            c_w_and_t+=1
        if tup[1]==tag:
            c_t+=1
    return (c_w_and_t, c_t)

# Emission Probabilities can be calculated at the time of sentence processing
# :)
