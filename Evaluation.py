

import nltk
data=nltk.corpus.brown.tagged_sents(tagset="universal")

from sklearn.model_selection import train_test_split
train_data,test_data=train_test_split(data, test_size=0.2, train_size=0.8, random_state=80)

y_actual=test_data

y_predicted=test_data

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

def evaluate_model(y_actual, y_predicted):
    tag_seq_act=[]
    tag_seq_pred=[]
    uniq_tag=set()
    uniq_tag_dict={}
    for sent in y_actual:
        for tup in sent:
            tag_seq_act.append(tup[1])
            uniq_tag.add(tup[1])
    uniq_tag=list(uniq_tag)
   
    for i in range(len(uniq_tag)):
        uniq_tag_dict[uniq_tag[i]]=i

    for sent in y_predicted:
        for tup in sent:
            tag_seq_pred.append(tup[1])
            
    for i,tag in enumerate(tag_seq_act):
        tag_seq_act[i]=uniq_tag_dict[tag]
        
    for i,tag in enumerate(tag_seq_pred):
        tag_seq_pred[i]=uniq_tag_dict[tag]

    print(uniq_tag_dict)
    matched_tags=0
    for i in range(len(tag_seq_act)):
        if tag_seq_act[i]==tag_seq_pred[i]:
            matched_tags+=1
    precision=matched_tags/len(tag_seq_act)
    recall=matched_tags/len(tag_seq_pred)
    F1_score=(2*precision*recall)/(precision+recall)
    conf_matrix=confusion_matrix(tag_seq_act,tag_seq_pred)
    return [precision, recall, F1_score, conf_matrix, uniq_tag]

[prec,rec,f1,conf,label] = evaluate_model(y_actual,y_predicted)


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

create_conf_matrix(conf, label)

