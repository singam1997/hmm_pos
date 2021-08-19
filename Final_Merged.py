#!/usr/bin/env python
# coding: utf-8

# In[70]:


import numpy as np
import pandas as pd
import itertools
import matplotlib
import nltk
import random
import math
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

#Brown-Corpus
nltk.download('brown')                                   
from nltk.corpus import brown

#Use universal tagset , currently not using
sentence_tag = brown.tagged_sents(tagset="universal")
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
        
for sent in test_dataset:
  for (word,tag) in sent:
    word=word.lower()
    try:
      if tag not in tags_of_tokens[word]:
        tags_of_tokens[word].append(tag)
    except:
      list_of_tags = []
      list_of_tags.append(tag)
      tags_of_tokens[word] = list_of_tags
# #Each word and its corresponding tags in the test dataset

#Seperating the test data into test words and test tags
test_words=[]
test_tags=[]
for sent in test_dataset:
  temp_word=[]
  temp_tag=[]
  for (word,tag) in sent:
    temp_word.append(word.lower())
    temp_tag.append(tag)
  test_words.append(temp_word)
  test_tags.append(temp_tag)

#Viterbi Algorithm Implementation
predicted_tags = []                #Final list for prediction
for i in range(len(test_words)):   # for each tokenized sentence in the test data (test_words is a list of lists)
  sent = test_words[i]
  #storing_values is a dictionary which stores the required values
  #ex: storing_values = {step_no.:{state1:[previous_best_state,value_of_the_state]}}                
  storing_values = {}              
  for q in range(len(sent)):
    step = sent[q]
    #for the starting word of the sentence
    if q == 1:                
      storing_values[q] = {}
      tags = tags_of_tokens[step]
      for t in tags:
        #this is applied since we do not know whether the word in the test data is present in train data or not
        try:
          storing_values[q][t] = ['^^',bigram_tag_prob['^^'][t]*train_emission_prob[t][step]]
        #if word is not present in the train data but present in test data we assign a very low probability of 0.0001
        except:
          storing_values[q][t] = ['^^',0.0001]
    
    #if the word is not at the start of the sentence
    if q>1:
      storing_values[q] = {}
      previous_states = list(storing_values[q-1].keys())   # loading the previous states
      current_states  = tags_of_tokens[step]               # loading the current states
      #calculation of the best previous state for each current state and then storing
      #it in storing_values
      for t in current_states:                             
        temp = []
        for pt in previous_states:                         
          try:
            temp.append(storing_values[q-1][pt][1]*bigram_tag_prob[pt][t]*train_emission_prob[t][step])
          except:
            temp.append(storing_values[q-1][pt][1]*0.0001)
        max_temp_index = temp.index(max(temp))
        best_pt = previous_states[max_temp_index]
        storing_values[q][t]=[best_pt,max(temp)]

  #Backtracing to extract the best possible tags for the sentence
  pred_tags = []
  total_steps_num = storing_values.keys()
  last_step_num = max(total_steps_num)
  for bs in range(len(total_steps_num)):
    step_num = last_step_num - bs
    if step_num == last_step_num:
      pred_tags.append('$$')
      pred_tags.append(storing_values[step_num]['$$'][0])
    if step_num<last_step_num and step_num>0:
      pred_tags.append(storing_values[step_num][pred_tags[len(pred_tags)-1]][0])
  predicted_tags.append(list(reversed(pred_tags)))




# #Calculating the accuracy based on tagging each word in the test data.
# right = 0 
# wrong = 0
# for i in range(len(test_tags)):
#   gt = test_tags[i]
#   pred = predicted_tags[i]
#   for h in range(len(gt)):
#     if gt[h] == pred[h]:
#       right = right+1
#     else:
#       wrong = wrong +1 

# print('Accuracy on the test data is: ',right/(right+wrong))
# print('Loss on the test data is: ',wrong/(right+wrong))
tag_seq_act=[]
tag_seq_pred=[]
uniq_tag=set()
uniq_tag_dict={}
for li in test_tags:
    for tag in li:
        tag_seq_act.append(tag)

for li in predicted_tags:
    for tag in li:
        tag_seq_pred.append(tag)
        
#Removing $$ and END from the Model Evaluation
for item in tag_seq_act:
    if item=='$$':
        tag_seq_act.remove(item)
        
for item in tag_seq_pred:
    if item=='$$':
        tag_seq_pred.remove(item)

for item in tag_seq_act:
    if item=='^^':
        tag_seq_act.remove(item)
        
for item in tag_seq_pred:
    if item=='^^':
        tag_seq_pred.remove(item)

        
#Evaluating the Model
for tag in tag_seq_act:
    uniq_tag.add(tag)

uniq_tag=list(uniq_tag)
   
for i in range(len(uniq_tag)):
    uniq_tag_dict[uniq_tag[i]]=i
            
for i,tag in enumerate(tag_seq_act):
    tag_seq_act[i]=uniq_tag_dict[tag]
        
for i,tag in enumerate(tag_seq_pred):
    tag_seq_pred[i]=uniq_tag_dict[tag]


matched_tags=0
for i in range(len(tag_seq_act)):
    if tag_seq_act[i]==tag_seq_pred[i]:
        matched_tags+=1
precision=matched_tags/len(tag_seq_act)
recall=matched_tags/len(tag_seq_pred)
F1_score=(2*precision*recall)/(precision+recall)
F_point_5=(1.25*precision*recall)/((0.25*precision)+recall)
F2_score=(5*precision*recall)/((4*precision)+recall)
conf_matrix=confusion_matrix(tag_seq_act,tag_seq_pred)
# return [precision, recall, F1_score, conf_matrix, uniq_tag]

print("Precision:",precision*100)
print("Recall:",recall*100)
print("F1 Score:",F1_score)
print("F0.5 Score:",F_point_5)
print("F2 Score:",F2_score)

def create_conf_matrix(conf, labels):
    font = {'family' : 'DejaVu Sans',
    'weight' : 'bold',
    'size'   : 22}
    plt.rc('font', **font)
    cm_obj=ConfusionMatrixDisplay(conf, display_labels=labels)
    fig, ax = plt.subplots(figsize=(30,30))
    ax.set_xticklabels(labels,fontsize=20)
    ax.set_yticklabels(labels,fontsize=20)
    cm_obj.plot(ax=ax)
    cm_obj.ax_.set(
            title="Confusion Matrix",
            xlabel="Predicted",
            ylabel="Actual"
    )
    ax.xaxis.label.set_size(30)
    ax.yaxis.label.set_size(30)
    ax.title.set_size(30)

def per_POS_evaluation(conf_matrix,uniq_tag):
    li=[]
    for i in range(len(conf_matrix)):
        rt,ct=0,0
        for j in range(len(conf_matrix)):
            rt+=conf_matrix[i][j]
            ct+=conf_matrix[j][i]
        A=conf_matrix[i][i]
        prec=A/ct
        rec=A/rt
        F1=(2*prec*rec)/(prec+rec)
        li.append([prec,rec,F1])
    di={}
    i=0
    for l in li:
        di[uniq_tag[i]]=l
        i+=1
    table=pd.DataFrame.from_dict(di, orient='index')
    table.columns=['Precision', 'Recall', 'F1_Score']
    display(table)
    

create_conf_matrix(conf_matrix, uniq_tag)

per_POS_evaluation(conf_matrix, uniq_tag)


# In[69]:


display(table)


# In[50]:


import pandas as pd

li=[]
for i in range(len(conf_matrix)):
    rt,ct=0,0
    for j in range(len(conf_matrix)):
        rt+=conf_matrix[i][j]
        ct+=conf_matrix[j][i]
    A=conf_matrix[i][i]
    prec=A/ct
    rec=A/rt
    F1=(2*prec*rec)/(prec+rec)
    li.append([prec,rec,F1])

di={}
i=0
for l in li:
    di[uniq_tag[i]]=l
    i+=1
    
table2=pd.DataFrame.from_dict(di, orient='index')
table2.columns=['Precision', 'Recall', 'F1_Score']


# In[51]:


table2


# In[53]:


tag_seq_act.remove('END')


# In[54]:


for i in range(len(tag_set_act))


# In[57]:


for i in range(len(tag_seq_act)):
    if tag_seq_act[i]=='END':
        tag_seq_act.remove('END')


# In[59]:


c=0
for i in range(len(tag_seq_act)):
    if tag_seq_act[i]=='END':
        c+=1


# In[60]:


c


# In[ ]:




