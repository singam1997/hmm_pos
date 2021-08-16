# -*- coding: utf-8 -*-
"""MergePart1_NLPass1.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1k3M758YWhbjLWl3Hi2zzI07SbvQaPuKT
"""

import numpy as np
import itertools
import matplotlib
import nltk
import random
import math

#Brown-Corpus
nltk.download('brown')                                   
from nltk.corpus import brown

#Use universal tagset , currently not using
sentence_tag = brown.tagged_sents()
modified_sentence_tag=[]
for sent in sentence_tag:

  sent.insert(0,('^^','^^'))         # Sentence starts with '^^'
  sent.append(('$$','$$'))           # Sentence ends with '$$'
  modified_sentence_tag.append(sent)

#Shuffle the whole corpus
random.shuffle(modified_sentence_tag)

#Divide corpus into 5 equal parts
sentences_set1=modified_sentence_tag[:math.floor(len(modified_sentence_tag)*1/5)]
sentences_set2=modified_sentence_tag[math.floor(len(modified_sentence_tag)*1/5):math.floor(len(modified_sentence_tag)*2/5)]
sentences_set3=modified_sentence_tag[math.floor(len(modified_sentence_tag)*2/5):math.floor(len(modified_sentence_tag)*3/5)]
sentences_set4=modified_sentence_tag[math.floor(len(modified_sentence_tag)*3/5):math.floor(len(modified_sentence_tag)*4/5)]
sentences_set5=modified_sentence_tag[math.floor(len(modified_sentence_tag)*4/5):]

#For 5 Fold Cross Validation Train and test set
# Set1 as test set
train_sentences_1=sentences_set2+sentences_set3+sentences_set4+sentences_set5
test_sentences_1=sentences_set1

# Set2 as test set
train_sentences_2=sentences_set1+sentences_set3+sentences_set4+sentences_set5
test_sentences_2=sentences_set2

# Set3 as test set
train_sentences_3=sentences_set1+sentences_set2+sentences_set4+sentences_set5
test_sentences_3=sentences_set3

# Set4 as test set
train_sentences_4=sentences_set1+sentences_set2+sentences_set3+sentences_set5
test_sentences_4=sentences_set4

# Set5 as test set
train_sentences_5=sentences_set1+sentences_set2+sentences_set3+sentences_set4
test_sentences_5=sentences_set5

#For now only one set(set5) out of all sets is used, later on after the parts are merged we need to calculate probabilities, use viterbi and analyze on other sets as well
train_dataset = train_sentences_5
test_dataset = test_sentences_5

#Creation of a dictionary whose keys are tags and values contain words which have correspoding tag in the taining dataset
#example:- 'TAG':{word1: count(word1,'TAG')} count(word1,'TAG') means how many times the word is tagged as 'TAG'
train_word_tag = {}
for sent in train_dataset:
  for (word,tag) in sent:
    word=word.lower()            # removing ambiguity from capital letters 
    try:
      try:
        train_word_tag[tag][word]+=1
      except:
        train_word_tag[tag][word]=1
    except:
        train_word_tag[tag]={word:1}

#Calculation of emission probabilities using train_word_tag
train_emission_prob={}
for key in train_word_tag.keys():
  train_emission_prob[key]={}
  count = sum(train_word_tag[key].values())                           # count is total number of words tagged as a 'TAG'
  for key2 in train_word_tag[key].keys():
    train_emission_prob[key][key2]=train_word_tag[key][key2]/count    
#Emission probability is #times a word occured as 'TAG' / total number of 'TAG' words
#example: number of times 'Sandeep' occured as Noun / total number of nouns

#Estimating the bigrams of tags to be used for calculation of transition probability 
#Bigram Assumption is made, the current tag depends only on the previous tag
bigram_tag_data = {}
for sent in train_dataset:
  bi=list(nltk.bigrams(sent))
  for b1,b2 in bi:
    try:
      try:
        bigram_tag_data[b1[1]][b2[1]]+=1
      except:
        bigram_tag_data[b1[1]][b2[1]]=1
    except:
      bigram_tag_data[b1[1]]={b2[1]:1}
#bigram_tag_data is storing the values for every tag.
#Every key is a tag and value is tag followed for that key and corresponding counts.
#example: how many times an adj is followed by a noun {Noun:{Adj:3}}, here its 3 times.

#Calculation of the probabilities of tag bigrams for transition probability
#We already made a bigram assumption  
#Also note that since we are also considering $, the $ row of transition probability matrix give us the initial probabilities as well
bigram_tag_prob={}
for key in bigram_tag_data.keys():
  bigram_tag_prob[key]={}
  count=sum(bigram_tag_data[key].values())              # count is total number of times a 'TAG' has occured
  for key2 in bigram_tag_data[key].keys():
    bigram_tag_prob[key][key2]=bigram_tag_data[key][key2]/count
#Tranmission probability is #times a TAG2 is preceded by TAG1 / total number of times TAG1 exists in dataset
#example: number of times a noun occured before adjective / total number of times a noun occurred

#Calculation the possible tags for each word in the entire daatset
#Note: Here we have used the whole data(Train dataset + Test dataset)
#Reason: Words present in Test data is not subset pf Train data.
#The above thing can be neglected if not necessay, but it improves our accuracy of the model 
tags_of_tokens = {}
count=0
for sent in train_dataset:
  for (word,tag) in sent:
    word=word.lower()
    try:
      if tag not in tags_of_tokens[word]:
        tags_of_tokens[word].append(tag)
    except:
      list_of_tags = []
      list_of_tags.append(tag)
      tags_of_tokens[word] = list_of_tags
#Each word and its corresponding tags in the train dataset
        
# for sent in test_dataset:
#   for (word,tag) in sent:
#     word=word.lower()
#     try:
#       if tag not in tags_of_tokens[word]:
#         tags_of_tokens[word].append(tag)
#     except:
#       list_of_tags = []
#       list_of_tags.append(tag)
#       tags_of_tokens[word] = list_of_tags
# #Each word and its corresponding tags in the test dataset