# -*- coding: utf-8 -*-
# Importing necessary libraries

import numpy as np
import itertools
import matplotlib
import nltk
import random
import math
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import statistics
import pandas as pd

# display_conf_matrix() takes the confusion matrix and produces a visual of it
def display_conf_matrix(conf, labels):
  # Set font values for the confusion matrix image
    font = {'family' : 'DejaVu Sans',
    'weight' : 'bold',
    'size'   : 10}
    plt.rc('font', **font)
    cm_obj=ConfusionMatrixDisplay(conf, display_labels=labels)
    fig, ax = plt.subplots(figsize=(30,30))
    ax.set_xticklabels(labels,fontsize=9)
    ax.set_yticklabels(labels,fontsize=9)
    # values_format='' is used to avoid showing in precision format eg 4200 shoudn't be shown as 4.2e3 (just to make the confusion matrix look better) 
    cm_obj.plot(ax=ax,values_format='')
    # set the axis labels and title
    cm_obj.ax_.set(
            title="Confusion Matrix",
            xlabel="Predicted",
            ylabel="Actual"
    )
    ax.xaxis.label.set_size(30)
    ax.yaxis.label.set_size(30)
    ax.title.set_size(30)


# per_POS_evaluation() calculates the per POS performance
# For each pair for tags(one from row and other column) in confusion matrix
# calculate the precision, recall and F1 Score
# All these scores are stored in a dataframe and the dataframe is returned
def per_POS_evaluation(conf_matrix,uniq_tag):
    li=[]
    for i in range(len(conf_matrix)):
        rt,ct=0,0
        # rt= no. of actual tags for tag i
        # rt= no. of predicted(obtained) tags for tag i
        for j in range(len(conf_matrix)):
            rt+=conf_matrix[i][j]
            ct+=conf_matrix[j][i]
        A=conf_matrix[i][i]
        # Calculate scores for each POS tag 
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
    return table

# Getting the brown corpus
print("Downloading files from NLTK please wait...")
nltk.download('all', quiet=True)
print("NLTK files downloaded!")                 
from nltk.corpus import brown

# Sentences fetched with its word tagged with universal tagset and sentences are
# prefixed and suffixed by delimiters. If originally sentence contains a 
# delimiter then set it as ('<DEL>','X')
sentence_tag = nltk.corpus.brown.tagged_sents(tagset="universal")
modified_sentence_tag=[]
for sent in sentence_tag:
  for word,tag in sent:
    if word=='^^' or word=='$$':
      word='<DEL>'
      tag='X'
  sent.insert(0,('^^','^^'))         # Sentence starts with '^^'
  sent.append(('$$','$$'))           # Sentence ends with '$$'
  modified_sentence_tag.append(sent)

# Shuffle the whole corpus uniformly
random.shuffle(modified_sentence_tag)

# Divide corpus into 5 equal parts
sentences_set1=modified_sentence_tag[:math.floor(len(modified_sentence_tag)*1/5)]
sentences_set2=modified_sentence_tag[math.floor(len(modified_sentence_tag)*1/5):math.floor(len(modified_sentence_tag)*2/5)]
sentences_set3=modified_sentence_tag[math.floor(len(modified_sentence_tag)*2/5):math.floor(len(modified_sentence_tag)*3/5)]
sentences_set4=modified_sentence_tag[math.floor(len(modified_sentence_tag)*3/5):math.floor(len(modified_sentence_tag)*4/5)]
sentences_set5=modified_sentence_tag[math.floor(len(modified_sentence_tag)*4/5):]

train_sentences=[[],[],[],[],[]]
test_sentences=[[],[],[],[],[]]

# For 5 Fold Cross Validation Train and test set
# Set1 as test set
train_sentences[0]=sentences_set2+sentences_set3+sentences_set4+sentences_set5
test_sentences[0]=sentences_set1

# Set2 as test set
train_sentences[1]=sentences_set1+sentences_set3+sentences_set4+sentences_set5
test_sentences[1]=sentences_set2

# Set3 as test set
train_sentences[2]=sentences_set1+sentences_set2+sentences_set4+sentences_set5
test_sentences[2]=sentences_set3

# Set4 as test set
train_sentences[3]=sentences_set1+sentences_set2+sentences_set3+sentences_set5
test_sentences[3]=sentences_set4

# Set5 as test set
train_sentences[4]=sentences_set1+sentences_set2+sentences_set3+sentences_set4
test_sentences[4]=sentences_set5

precision_sets=[0]*5
recall_sets=[0]*5
F1_score_sets=[0]*5
F05_score_sets=[0]*5
F2_score_sets=[0]*5
pos_estimation_sets=[pd.DataFrame]*5
print("Running Viterbi over all 5 cross validation sets...") #Note this is just for showing progress, viterbi is executed further in the loop
for setno in range(5):
  # For each set calculate the transmission and emission probabilities on training set
  # And perform viterbi on test set. Later find per POS and overall estimation
  train_dataset = train_sentences[setno]
  test_dataset = test_sentences[setno]

  ## EMISSION PROBABILITY TABLE
  # Creation of a dictionary whose keys are tags and values contain words which have corresponding tag in the taining dataset
  # example:- 'TAG':{word1: count(word1,'TAG')} count(word1,'TAG') means how many times the word is tagged as 'TAG'
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

  ## TRANSITION PROBABILITY TABLE
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

  #Calculation the possible tags for each word in the train dataset
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

  # Getting words and their corresponding tags from the test set
  # Seperating the test data into test words and test tags
  test_words=[]
  test_tags=[]
  for sent in test_dataset:
    temp_word=[]
    temp_tag=[]
    for (word,tag) in sent:
      temp_word.append(word.lower()) # words of a sentence in test dataset
      temp_tag.append(tag) # tags of a sentence in test dataset
    test_words.append(temp_word) # list with words of a sentence(tokenized sentence) appended to a list of list
    test_tags.append(temp_tag) # list with tags of a sentence(tokenized sentence) appended to a list of list

  #VITERBI ALGORITHM IMPLEMENTATION
# For each word in a sentence
# – The probability of best candidates for each tag at previous level is
#   multiplied with the emission probability and the transition
#   probability of possible tags based on current word
# – The best candidate for each tag at the current level is chosen and
#   the previous tag is kept a track of for backtracking
# – The result of forward propagation at each level looks like the
#   following
#   LEVEL_K:{TAG1:{best_candidate_among_Tag1,probability_of the
#   best candidate} , {TAG2:{previous_tag_of_best_candidate_
#   among_Tag2, probability_of the best candidate}, ...}
# – When backtracking start from the end to start choosing the
#   predicted tag based on LEVEL information and the predicted tag
#   that follows the word.
# – In case an unseen word arrives the next tag is set to ‘NOUN’ and
#   the emission probability is set to a low probability (0.0001). This is
#   based on the observation from the corpus that ≈ 63% of the
#   unseen words were noun.
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
        try:
          tags = tags_of_tokens[step]
        except:
          # print(step,test_tags_of_tokens[step])
          tags=['NOUN'] #tags_of_unseen_tokens
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
        try:
          current_states  = tags_of_tokens[step]               # loading the current states
        except:
          current_states = ['NOUN']#tags_of_unseen_tokens
        #calculation of the best previous state for each current state and then storing
        #it in storing_values
        for t in current_states:                             
          temp = []
          for pt in previous_states:                         
            try:
              temp.append(storing_values[q-1][pt][1]*bigram_tag_prob[pt][t]*train_emission_prob[t][step]) # If seen word
            except:
              temp.append(storing_values[q-1][pt][1]*0.0001)
          max_temp_index = temp.index(max(temp))
          best_pt = previous_states[max_temp_index]
          storing_values[q][t]=[best_pt,max(temp)] #Store the best previous tag for each best candidate per tag and the meximum probability
 
    #Backtracing to extract the best possible tags for the sentence
    # for each word looking the current word and the word and tag next to it in the sentence backtrack
    # to get the tag of current word
    pred_tags = [] #predicted tags by viterbi using backtracking
    total_steps_num = storing_values.keys()
    last_step_num = max(total_steps_num)     # Begin from the last word which will end the delimiter
    for bs in range(len(total_steps_num)):    
      step_num = last_step_num - bs          
      if step_num == last_step_num:
        pred_tags.append('$$')
        pred_tags.append(storing_values[step_num]['$$'][0]) 
      if step_num<last_step_num and step_num>0:
        pred_tags.append(storing_values[step_num][pred_tags[len(pred_tags)-1]][0]) #Looking into storing value fetch the best previous tag for the current word
    predicted_tags.append(list(reversed(pred_tags)))


#Now that the tags are predicted, get the actual and predicted tags so that analysis can be done
  tag_seq_act=[]
  tag_seq_pred=[]
  uniq_tag=set()
  uniq_tag_dict={}
  for li in test_tags:
      for tag in li:
          if(tag!="^^" and tag!="$$"): #Exclude delimiters
            tag_seq_act.append(tag)

  for li in predicted_tags:
      for tag in li:
          if(tag!="^^" and tag!="$$"): #Exclude delimiters
            tag_seq_pred.append(tag)
      
  for tag in tag_seq_act:
      uniq_tag.add(tag)

  uniq_tag=list(uniq_tag)
    
  for i in range(len(uniq_tag)):
      uniq_tag_dict[uniq_tag[i]]=i
              
  for i,tag in enumerate(tag_seq_act):
      tag_seq_act[i]=uniq_tag_dict[tag]

  for i,tag in enumerate(tag_seq_pred):
      tag_seq_pred[i]=uniq_tag_dict[tag]

# Calculate the precision, Recall and F-Scores by comparing the actual and predicted tags
  matched_tags=0
  for i in range(len(tag_seq_act)):
      if tag_seq_act[i]==tag_seq_pred[i]:
        matched_tags+=1
  # Estimations for the current set
  precision=matched_tags/(len(tag_seq_pred))
  recall=matched_tags/(len(tag_seq_act))
  F1_score=(2*precision*recall)/(precision+recall)
  F05_score=(1.25*precision*recall)/(0.25*precision+recall)
  F2_score=(5*precision*recall)/(4*precision+recall)
  # A confusion matrix is created which is used to diplay the confusion matrix and for per pos evaluation
  conf_matrix=confusion_matrix(tag_seq_act,tag_seq_pred)
  pos_estimation=per_POS_evaluation(conf_matrix, uniq_tag)

  #Store every set's estimations
  precision_sets[setno]=precision
  recall_sets[setno]=recall
  F1_score_sets[setno]=F1_score
  F05_score_sets[setno]=F05_score
  F2_score_sets[setno]=F2_score
  pos_estimation_sets[setno]=pos_estimation
  print("Set ",setno+1,"✓")

# After getting every set's estimations combine them
# For k fold cross validation
# mean(each of the k set estimations) ± standard_error(each of the k set estimations)
# standard_error= standard_deviation/square_root(k)
# example Overall precision=(Σ precision_set i)/5 ± squareroot([Σ(precision_set_i-precision_mean)²]/5) here 5 for 5 fold cross validation
print("===================\nOVERALL ESTIMATIONS\n===================")
print("Overall Precision:", "{:.6f}".format(statistics.mean(precision_sets)),"±","{:.6f}".format(statistics.stdev(precision_sets)/math.sqrt(setno+1)))
print("Overall Recall:" ,"{:.6f}".format(statistics.mean(recall_sets)),"±","{:.6f}".format(statistics.stdev(recall_sets)/math.sqrt(setno+1)))
print("Overall F1 Score:", "{:.6f}".format(statistics.mean(F1_score_sets)),"±","{:.6f}".format(statistics.stdev(F1_score_sets)/math.sqrt(setno+1)))
print("Overall F0.5 Score:", "{:.6f}".format(statistics.mean(F1_score_sets)),"±","{:.6f}".format(statistics.stdev(F1_score_sets)/math.sqrt(setno+1)))
print("Overall F2 Score:", "{:.6f}".format(statistics.mean(F1_score_sets)),"±","{:.6f}".format(statistics.stdev(F1_score_sets)/math.sqrt(setno+1)))
mean_POS_est=(pos_estimation_sets[0]+pos_estimation_sets[1]+pos_estimation_sets[2]+pos_estimation_sets[3]+pos_estimation_sets[4])/5
se_POS_est=np.sqrt((pos_estimation_sets[0]-mean_POS_est)**2+(pos_estimation_sets[1]-mean_POS_est)**2+(pos_estimation_sets[2]-mean_POS_est)**2+(pos_estimation_sets[3]-mean_POS_est)**2+(pos_estimation_sets[4]-mean_POS_est)**2)/5
print("\n===============\nPOS ESTIMATIONS\n===============\n++++\nMEAN\n++++\n",mean_POS_est,"\n++++++++++++++\nSTANDARD ERROR\n++++++++++++++\n",se_POS_est)

print("\n\nThe confusion matrix of set ",setno+1," will be displayed")
#Here confusion matrix for the set 5 is shown
display_conf_matrix(conf_matrix, uniq_tag)
#Following line is used to show the confusion matrix when the code is run on terminal, can be commented if using jupyter notebooks 
matplotlib.pyplot.show()
