import numpy as np
import itertools
import matplotlib
import nltk

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

#Dividing into train and test dataset
data_split_number = int(len(modified_sentence_tag)*0.8) # 80% train dataset and 20% test dataset
train_dataset = modified_sentence_tag[0:data_split_number]
test_dataset = modified_sentence_tag[data_split_number:]

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
#Each word and its corresponding tags in the test dataset



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




#Calculating the accuracy based on tagging each word in the test data.
right = 0 
wrong = 0
for i in range(len(test_tags)):
  gt = test_tags[i]
  pred = predicted_tags[i]
  for h in range(len(gt)):
    if gt[h] == pred[h]:
      right = right+1
    else:
      wrong = wrong +1 

print('Accuracy on the test data is: ',right/(right+wrong))
print('Loss on the test data is: ',wrong/(right+wrong))