"""
evaluate_taggers

Scott Harrison - 2018-08-10

Code to evaluate the Modern Greek POS taggers developed.

This process evaluates the various Modern Greek taggers based on the INTERA 
corpus. These taggers span simple (2 letter) tags for sequential, 
classification and deep learning, as well as more complex (4 letter) tags. The
taggers are developed and stored in the following scripts:
    - investigate_tagger_nltk_sequential
    - investigate_tagger_nltk_calssification
    - investigate_tagger_keras_deeplearning
    - investigate_tagger_complextags
    
All model finetuning and selection was performed using the val_sents data. The
evaluation here will make use of the previously unseen test_sents for INTERA
as well as the test data available for UDGreek and Tagged_Texts. Val_sents for 
INTERA will also be used here, but merely for verification purposes - ie to
align with the investigation results.

"""

import matplotlib.pyplot as plt
import pandas as pd
from greek_nlp_corpus import read_corpus
from greek_nlp_pos import (tic, toc, untag, compute_metrics, compare_results,
                           load_tagger, deep_learning_tag, compute_sent_acc,
                           save_tagger, add_features)

# set corpus and resource locations
CORPUS_DIR = './data/corpus/'
RESOURCES_DIR = './data/resources/'
RESULTS_DIR = './data/results/'


"""
# =============================================================================
# evaluate the simple tags
# =============================================================================
"""

# set user constants
TAG_LENGTH = 2
PROPORTION = 100


""" 0. set up the corpora for evaluation of tagging methods """

# load all corpus as tagged sentences
_, val_70, _ = read_corpus('INTERA', role='train',
                           proportion=70, tag_length=TAG_LENGTH)
_, val_100, test_int = read_corpus('INTERA', role='train',
                                   proportion=PROPORTION,
                                   tag_length=TAG_LENGTH)
_, _, test_ud = read_corpus('UDGreek')
_, _, test_tt = read_corpus('tagged_texts')


""" 1. sequential tagger """
seq_eval = dict()
seq_tag = load_tagger(RESOURCES_DIR+'Greek_POS_seq.pkl')
# word level
seq_eval['verification'] = seq_tag.evaluate(val_70)
tic()
seq_eval['evaluate'] = seq_tag.evaluate(test_int)
seq_eval['evaluate_time'] = toc()
seq_eval['ud_greek'] = seq_tag.evaluate(test_ud)
seq_eval['tagged_text'] = seq_tag.evaluate(test_tt)
# sentence level
pred_int = [seq_tag.tag(s) for s in untag(test_int)]
seq_eval['sent_evaluate'] = compute_sent_acc(test_int, pred_int)
pred_ud = [seq_tag.tag(s) for s in untag(test_ud)]
seq_eval['sent_ud_greek'] = compute_sent_acc(test_ud, pred_ud)
pred_tt = [seq_tag.tag(s) for s in untag(test_tt)]
seq_eval['sent_tagged_text'] = compute_sent_acc(test_tt, pred_tt)
print('\n')
print(seq_eval)


""" 2. classification tagger """
class_eval = dict()
class_tag = load_tagger(RESOURCES_DIR+'Greek_POS_class.pkl')
# word level
class_eval['verification'] = class_tag.evaluate(val_70)
tic()
class_eval['evaluate'] = class_tag.evaluate(test_int)
class_eval['evaluate_time'] = toc()
class_eval['ud_greek'] = class_tag.evaluate(test_ud)
class_eval['tagged_text'] = class_tag.evaluate(test_tt)
# sentence level
pred_int = [class_tag.tag(s) for s in untag(test_int)]
class_eval['sent_evaluate'] = compute_sent_acc(test_int, pred_int)
pred_ud = [class_tag.tag(s) for s in untag(test_ud)]
class_eval['sent_ud_greek'] = compute_sent_acc(test_ud, pred_ud)
pred_tt = [class_tag.tag(s) for s in untag(test_tt)]
class_eval['sent_tagged_text'] = compute_sent_acc(test_tt, pred_tt)
print('\n')
print(class_eval)


""" 3. deep learning """
deep_eval = dict()
# word level
tag_val = deep_learning_tag(untag(val_100), 'Greek_POS_DL', RESOURCES_DIR)
_, deep_eval['verification'] = compute_metrics(val_100, tag_val)
tic()
tag_int = deep_learning_tag(untag(test_int), 'Greek_POS_DL', RESOURCES_DIR)
_, deep_eval['evaluate'] = compute_metrics(test_int, tag_int)
deep_eval['evaluate_time'] = toc()
tag_ud = deep_learning_tag(untag(test_ud), 'Greek_POS_DL', RESOURCES_DIR)
_, deep_eval['ud_greek'] = compute_metrics(test_ud, tag_ud)
tag_tt = deep_learning_tag(untag(test_tt), 'Greek_POS_DL', RESOURCES_DIR)
_, deep_eval['tagged_text'] = compute_metrics(test_tt, tag_tt)
# sentence level
deep_eval['sent_evaluate'] = compute_sent_acc(test_int, tag_int)
deep_eval['sent_ud_greek'] = compute_sent_acc(test_ud, tag_ud)
deep_eval['sent_tagged_text'] = compute_sent_acc(test_tt, tag_tt)
print('\n')
print(deep_eval)


""" 4. save original tagset results """
seq_df = pd.DataFrame.from_dict(seq_eval, orient='index')
seq_df.columns = ['NLTK Sequence']
clas_df = pd.DataFrame.from_dict(class_eval, orient='index')
clas_df.columns = ['NLTK Classification']
deep_df = pd.DataFrame.from_dict(deep_eval, orient='index')
deep_df.columns = ['Keras Deep Learning']
basic_df = seq_df.join(clas_df).join(deep_df)
save_tagger(RESULTS_DIR+'EVAL_basic.pkl', basic_df)


"""
# =============================================================================
# evaluate the complex tags
# =============================================================================
"""

# set user constants
TAG_LENGTH = 20
PROPORTION = 100


""" 0. set up the corpora for evaluation of tagging methods """

# load the corpus as tagged sentences
_, val_sents, test_sents = read_corpus('INTERA', 
                                       role='train',
                                       proportion=PROPORTION,
                                       tag_length=TAG_LENGTH)


# include case in tags
def add_tags(sents, tag_main, tag_add_ind):
    sents_upd = [[word if word[1][:2] != tag_main 
                  else (word[0], 
                        word[1][:2]+word[1][2*(tag_add_ind-1):(2*tag_add_ind)])
                  for word in sent] for sent in sents]
    return(sents_upd)
def rem_tags(sents, tag_main):
    sents_upd = [[word if word[1][:2] != tag_main 
                  else (word[0], word[1][:2])
                  for word in sent] for sent in sents]
    return(sents_upd)
def update_tags(sents):
    sents = add_tags(sents, 'PN', 6)
    sents = add_tags(sents, 'VB', 10)
    sents = add_tags(sents, 'AJ', 5)
    sents = add_tags(sents, 'NO', 5)
    sents = add_tags(sents, 'NM', 5)
    sents = add_tags(sents, 'AT', 5)
    sents = add_tags(sents, 'AS', 5)
    sents = rem_tags(sents, 'PU')
    sents = rem_tags(sents, 'DI')
    sents = rem_tags(sents, 'RG')
    #sents = rem_tags(sents, 'PT')
    sents = rem_tags(sents, 'AD')
    sents = rem_tags(sents, 'CJ')
    sents = rem_tags(sents, 'DA')
    sents = rem_tags(sents, 'IJ')
    sents = rem_tags(sents, 'AB')
    return(sents)
val_sents = update_tags(val_sents)
test_sents = update_tags(test_sents)


""" 1. sequential tagger """
seq_ctag_eval = dict()
seq_ctag_tag = load_tagger(RESOURCES_DIR+'Greek_POS_ctag_seq.pkl')
# word level
seq_ctag_eval['verification'] = seq_ctag_tag.evaluate(val_sents)
tic()
seq_ctag_eval['evaluate'] = seq_ctag_tag.evaluate(test_sents)
seq_ctag_eval['evaluate_time'] = toc()
# sentence level
test_pred = [seq_ctag_tag.tag(s) for s in untag(test_sents)]
seq_ctag_eval['sent_evaluate'] = compute_sent_acc(test_sents, test_pred)
print('\n')
print(seq_ctag_eval)


""" 2. classification tagger """
class_ctag_eval = dict()
class_ctag_tag = load_tagger(RESOURCES_DIR+'Greek_POS_ctag_class.pkl')
# word level
class_ctag_eval['verification'] = class_ctag_tag.evaluate(val_sents)
tic()
class_ctag_eval['evaluate'] = class_ctag_tag.evaluate(test_sents)
class_ctag_eval['evaluate_time'] = toc()
# sentence level
test_pred = [class_ctag_tag.tag(s) for s in untag(test_sents)]
class_ctag_eval['sent_evaluate'] = compute_sent_acc(test_sents, test_pred)
print('\n')
print(class_ctag_eval)


""" 3. deep learning """
deep_ctag_eval = dict()
# word level
val_pred = deep_learning_tag(untag(val_sents), 'Greek_POS_ctag_DL', RESOURCES_DIR)
_, deep_ctag_eval['verification'] = compute_metrics(val_sents, val_pred)
tic()
test_pred = deep_learning_tag(untag(test_sents), 'Greek_POS_ctag_DL', 
                            RESOURCES_DIR)
_, deep_ctag_eval['evaluate'] = compute_metrics(test_sents, test_pred)
deep_ctag_eval['evaluate_time'] = toc()
# sentence level
deep_ctag_eval['sent_evaluate'] = compute_sent_acc(test_sents, test_pred)
print('\n')
print(deep_ctag_eval)


""" 4. save finer tagset results """
seq_ctag_df = pd.DataFrame.from_dict(seq_ctag_eval, orient='index')
seq_ctag_df.columns = ['NLTK Sequence']
clas_ctag_df = pd.DataFrame.from_dict(class_ctag_eval, orient='index')
clas_ctag_df.columns = ['NLTK Classification']
deep_df = pd.DataFrame.from_dict(deep_ctag_eval, orient='index')
deep_df.columns = ['Keras Deep Learning']
complex_df = seq_ctag_df.join(clas_ctag_df).join(deep_df)
save_tagger(RESULTS_DIR+'EVAL_complex.pkl', complex_df)


"""
# =============================================================================
# review the evaluation results
# =============================================================================
"""

# load data if necessary
basic_df = load_tagger(RESULTS_DIR+'EVAL_basic.pkl')
complex_df = load_tagger(RESULTS_DIR+'EVAL_complex.pkl')

# plot all test corpus - words
all_words = basic_df.loc[['evaluate','ud_greek','tagged_text'],:]
all_words = all_words.append(complex_df.loc[['evaluate'],:])
all_words = all_words*100
fig1 = all_words.plot.bar()
fig1.plot([-0.5,3.5],[97,97],'r--',lw=2,zorder=0)
fig1.plot([-0.5,3.5],[93,93],'r--',lw=2,zorder=0)
fig1.set_xticklabels(['INTERA','UD Corpus','Tagged Text', 
                      'INTERA\n(complex tagset)'])
fig1.set_ylabel('Word Accuracy (%)')
fig1.legend(loc=4)

# plot all test corpus - sentences
all_sents = basic_df.loc[['sent_evaluate','sent_ud_greek','sent_tagged_text'],:]
all_sents = all_sents.append(complex_df.loc[['sent_evaluate'],:])
all_sents = all_sents*100
fig2 = all_sents.plot.bar()
fig2.plot([-0.5,3.5],[55,55],'r--',lw=2,zorder=0)
fig2.set_xticklabels(['INTERA','UD Corpus','Tagged Text', 
                      'INTERA\n(complex tagset)'])
fig2.set_ylabel('Sentence Accuracy (%)')
fig2.legend(loc=9)

# plot accuracy v timing
basic_eval = basic_df.loc[['evaluate','evaluate_time'],:].T
complex_eval = complex_df.loc[['evaluate','evaluate_time'],:].T
plt.plot(basic_eval['evaluate']*100, basic_eval['evaluate_time'], 'ro', 
         label='INTERA - Simple Tags')
plt.plot(complex_eval['evaluate']*100, complex_eval['evaluate_time'], 'bo',
         label='INTERA - Complex Tags')
for label, x, y in zip(basic_eval.axes[0].tolist(), 
                       basic_eval['evaluate']*100, 
                       basic_eval['evaluate_time']):
    plt.annotate(label, xy=(x, y), xytext=(-10, 0),
        textcoords='offset points', ha='right', va='center')
for label, x, y in zip(basic_eval.axes[0].tolist(), 
                       complex_eval['evaluate']*100, 
                       complex_eval['evaluate_time']):
    plt.annotate(label, xy=(x, y), xytext=(10, 0),
        textcoords='offset points', ha='left', va='center')
plt.xlabel('Word Accuracy (%)')
plt.ylabel('Duration (s)')
plt.grid(zorder=0, color=[0.9,0.9,0.9])
plt.legend(loc=2)
plt.show()



