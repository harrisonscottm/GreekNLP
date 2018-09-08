"""
greek_nlp_pos

Scott Harrison - 2018-07-14

Functions to assist in training a POS tagger from a tagged corpus

This module contains a series of functions which assist in the training of a
POS tagger from a tagged corpus using NLTK and Keras deep learning.

"""

# module attributes - for inline examples
DEMO_DIR = './data/_demo_/'                        # location of demo resources

import time
import pickle
import pandas as pd
import matplotlib.pyplot as plt
from nltk import FreqDist
from keras.models import load_model
# used for inline examples only
from nltk.corpus import TaggedCorpusReader


""" 
# =============================================================================
# timing functions 
# =============================================================================
"""
global _tic_
def tic():                           # start timer and store in global variable
    global _tic_
    _tic_ = time.time()
def toc():                                  # calculate elapsed time and return
    _toc_ = time.time()-_tic_
    return(_toc_)


""" 
# =============================================================================
# untag function
# =============================================================================
"""
def untag(sents_tagged):
    sents_untagged = [[word[0] for word in sent] for sent in sents_tagged]
    return(sents_untagged)
    

""" 
# =============================================================================
# pickle tagger functions
# =============================================================================
"""
def load_tagger(file):
    pickle_file = open(file,'rb') 
    data = pickle.load(pickle_file)   
    pickle_file.close()
    return(data)
def save_tagger(file, data):
    pickle_file = open(file,'wb') 
    pickle.dump(data, pickle_file)   
    pickle_file.close()
    
  
""" 
# =============================================================================
# plot results function
# =============================================================================
"""
def plot_two_axis(xlab, x, y1lab, y1, y2lab, y2):
    g = [0.6, 0.6, 0.6]                                       # set grey colour
    fig, ax1 = plt.subplots()                                        # set axis
    ax1.set_xlabel(xlab)                                         # label x axis
    # plot and label main y axis
    ax1.plot(x, y1, 'k')
    ax1.set_ylabel(y1lab, color='k')
    ax1.tick_params('y', colors='k')
    # plot and label second y axis
    ax2 = ax1.twinx()
    ax2.plot(x, y2, color=g)
    ax2.set_ylabel(y2lab, color=g)
    ax2.tick_params('y', colors=g)
    # add grid and display
    fig.tight_layout()
    ax1.grid(True)
    plt.show()


""" 
# =============================================================================
# display the training metrics 
# =============================================================================
"""
def display_training_metrics(tagger_eval, decimals=1):
    """
    Displays the metrics store in the training evaluation variable.
    
    Given a training evaluation variable display the cleaned up metrics.
    
    >>> tag_eval = {'test_accuracy': 0.26, \
                    'train_time': 7.7808} 
    >>> display_training_metrics(tag_eval)
    Training Time: 7.8s
    Test Time    : Not provided
    Test Accuracy: 26.0%
    
    :param tagger_eval: The metrics for the tagger.
    :type  tagger_eval: Dict
    :param    decimals: Number of decimals to display
    :type     decimals: Integer
    :default  decimals: 1
    :rtype            : Display to screen
    :raise exception  : If a valid eval dictionary is not provided.
    """
    
    try:
        # extract the results
        if 'train_time' in tagger_eval:
            train_time = str(round(tagger_eval['train_time'],decimals)) + 's'
        else:
            train_time = 'Not provided'
        if 'test_time' in tagger_eval:
            test_time = str(round(tagger_eval['test_time'],decimals)) + 's'
        else:
            test_time = 'Not provided'
        if 'test_accuracy' in tagger_eval:
            test_accuracy = str(round(tagger_eval['test_accuracy']*100,
                                      decimals)) + '%'
        else:
            test_accuracy = 'Not provided'
        print('Training Time: ' + train_time + '\n' +
              'Test Time    : ' + test_time + '\n' +
              'Test Accuracy: ' + test_accuracy)
    except:
        # warn about invalid corpus
        return(print('ERR - display_training_metrics: ensure the training '
                     'metrics are valid.'))


""" 
# =============================================================================
# find most common items - words or tags
# =============================================================================
"""
def find_pos_most_common(sent_tag, item_type='word', item_n=100):
    """
    Function to find the most common tags or words in a corpus.
    
    Given a tagged corpus of sentences, this function finds the ITEM_N
    most common tags or words - depending on user input.
    
    >>> corp = TaggedCorpusReader(DEMO_DIR, r'DEMO_Corpus.txt')
    >>> find_pos_most_common(corp.tagged_sents(), item_type='tag', item_n=1)
    [('NO', 59)]
    
    :param      sent_tag: A tagged corpus of sentences.
    :type       sent_tag: A list of lists
    :param     item_type: Either 'tag' or 'word'
    :type      item_type: String
    :default   item_type: 'word'
    :param        item_n: The number of required items.
    :type         item_n: Integer
    :default      item_n: 100
    :rtype              : List
    :raise exception    : If an invalid item is provided.
    """
    
    # check for valid parameters
    if item_type not in ['word','tag']:
        return(print('ERR - find_most_common_item: ensure item is valid: '
              'item must be one of {''word'',''tag''}'))
    
    # get tagged words
    words_tagged = [wordtag for sent in sent_tag for wordtag in sent]
        
    # get required items
    if item_type == 'tag':
        items = [tag for (word,tag) in words_tagged]
    else:
        items = [word for (word,tag) in words_tagged]
        
    # get most frequent n
    items_freq = FreqDist(items).most_common(item_n)
    
    # return results
    return(items_freq)
        

"""
# =============================================================================
# create list of regular expressions
# =============================================================================
"""
def create_regexp_list(xlsx_file, file_dir):
    """
    Function to load an xlsx file of regular expressions for POS tagging.
    
    Given an xlsx file with three columns - REGEXP, EXAMPLE and POS - a list is 
    returned containing tuples of regex patterns and its POS tag.
    
    >>> create_regexp_list('DEMO_Regexp.xlsx', DEMO_DIR)
    [('^εναντίον$', 'AS'), ('^εξαιτίας$', 'AS'), ('^μεταξύ$', 'AS'), ('.+κύματα$', 'NO'), ('.+σιματα$', 'NO')]
    
    :param xlsx_file: A spreadsheet containing the tags and their regex.
    :type  xlsx_file: String.
    :param  file_dir: Location of the file.
    :type   file_dir: String.
    :rtype          : List.
    :raise exception: If XLSX_FILE does not refer to a valid file.
    """
    
    try:
        # load the POS tags spreadsheet
        pos_list = pd.read_excel(io=file_dir+xlsx_file)
    except:
        # warn about invalid file
        return(print('ERR - create_regex_list: ensure the filename provided '
                     'refers to a valid file.'))
    
    # remove extraneous information 
    del pos_list['example']
    # create a list of tuples
    pos_final = list(pos_list.itertuples(index=False, name=None))
    # return the list
    return(pos_final)
   
    
"""
# =============================================================================
# create deep learning obervations
# =============================================================================
"""
def create_observation(sent_tagged):
    """
    Function to create an observation for each word.
    
    Given a tagged corpus of sentences, this function creates one observation
    of features (X) and outputs (y) for each word.
    
    From 'Part-of-Speech tagging tutorial with the Keras Deep Learning library' 
    by Axel Bellec at becominghuman.ai.
    
    >>> sent = [[('σχεδιο', 'NO'), \
                 ('χαρτη', 'NO'), \
                 ('θεμελιωδων', 'NO'), \
                 ('δικαιωματων', 'NO'), \
                 ('της', 'AT'), \
                 ('ευρωπαϊκης', 'AJ'), \
                 ('ενωσης', 'NO')], \
                [('βρυξέλλες', 'NO'), \
                 (',', 'PU'), \
                 ('11', 'DI'), \
                 ('οκτωβρίoυ', 'NO'), \
                 ('2000', 'DI'), \
                 ('(', 'PU'), \
                 ('19.10', 'DI'), \
                 (')', 'PU')]]
    >>> X, y = create_observation(sent)
    >>> X[0]
    {'nb_terms': 7, 'term': 'σχεδιο', 'is_first': True, 'is_last': False, 'is_penultimate': False, 'is_capitalized': False, 'is_all_caps': False, 'is_all_lower': True, 'prefix-1': 'σ', 'prefix-2': 'σχ', 'prefix-3': 'σχε', 'suffix-1': 'ο', 'suffix-2': 'ιο', 'suffix-3': 'διο', 'prev2_word': '', 'prev2_tag': '', 'prev_word': '', 'prev_tag': '', 'next_word': 'χαρτη'}
    >>> y
    ['NO', 'NO', 'NO', 'NO', 'AT', 'AJ', 'NO', 'NO', 'PU', 'DI', 'NO', 'DI', 'PU', 'DI', 'PU']
    
    :param   sent_tagged: A corpus tokenised as sentences, tokenised as words.
    :type    sent_tagged: List of lists
    :rtype              : Two lists
    """
    
    # to store the results
    X, y = [], []
    for sent in sent_tagged:
        sent_notag = [word for (word,_) in sent]
        for index, (word, tag) in enumerate(sent):
            # add features for each sentence word
            X.append(add_features(sent_notag, index, []))
            y.append(tag)
    return(X, y)
    
    
"""
# =============================================================================
# create features from words in a sentence
# =============================================================================
"""
def add_features(sent_words, index, history):
    """
    Function to create features for a given word in a sentence.
    
    Given a sentence, this function creates a series of features for a given
    word.
    
    From 'Part-of-Speech tagging tutorial with the Keras Deep Learning library' 
    by Axel Bellec at becominghuman.ai.
    
    >>> sent = ['κείμενo', 'των', 'επεξηγήσεων', 'σχετικά', 'με','τo', \
                'πλήρες', 'κείμενo', 'τoυ', 'χάρτη', '.']
    >>> add_features(sent, 0, [])
    {'nb_terms': 11, 'term': 'κείμενo', 'is_first': True, 'is_last': False, 'is_penultimate': False, 'is_capitalized': False, 'is_all_caps': False, 'is_all_lower': True, 'prefix-1': 'κ', 'prefix-2': 'κε', 'prefix-3': 'κεί', 'suffix-1': 'o', 'suffix-2': 'νo', 'suffix-3': 'ενo', 'prev2_word': '', 'prev2_tag': '', 'prev_word': '', 'prev_tag': '', 'next_word': 'των'}

    :param    sent_words: A sentence tokenised as words.
    :type     sent_words: List
    :param         index: Index of the current word in the sentence
    :type          index: Integer
    :rtype              : Dict
    """
    
    # create empty history if no history given
    if len(history) == 0:
        history = ['']*len(sent_words)
    
    term = sent_words[index]
    return {
        'nb_terms': len(sent_words),
        'term': term.lower(),
        'is_first': index == 0,
        'is_last': index == len(sent_words)-1,
        'is_penultimate': index == len(sent_words)-2,
        'is_capitalized': term[:1].upper() == term[:1],
        'is_all_caps': term.upper() == term,
        'is_all_lower': term.lower() == term,
        'prefix-1': term[:1].lower(),
        'prefix-2': term[:2].lower(),
        'prefix-3': term[:3].lower(),
        'suffix-1': term[-1:].lower(),
        'suffix-2': term[-2:].lower(),
        'suffix-3': term[-3:].lower(),
        'prev2_word': '' if index < 2 else sent_words[index-2].lower(),
        'prev2_tag': '' if index < 2 else history[index-2],
        'prev_word': '' if index == 0 else sent_words[index-1].lower(),
        'prev_tag': '' if index == 0 else history[index-1],
        'next_word': '' if index == len(sent_words)-1 
                        else sent_words[index+1].lower(),
    }


"""
# =============================================================================
# plot deep learning model performance
# =============================================================================
"""
def plot_model_performance(model_hist):
    """
    Function to display the model performance during training.
    
    Given a the history of the training process display the performance over
    time (ie for each epoch).
    
    :param  model_hist: History output of model training.
    :type   model_hist: callbacks.History
    :rtype            : Two plots
    """ 
    
    # extract required values
    train_loss = model_hist.history.get('loss', [])
    train_accuracy = model_hist.history.get('acc', [])
    train_val_loss = model_hist.history.get('val_loss', [])
    train_val_accuracy = model_hist.history.get('val_acc', [])
    
    # plot model loss
    fig, (ax1, ax2) = plt.subplots(2, figsize=(10, 8))
    ax1.plot(range(1, len(train_loss) + 1), train_loss, color='k', label='training')
    ax1.plot(range(1, len(train_val_loss) + 1), train_val_loss, color='k', ls='--', label='validation')
    ax1.set_xlabel('')
    ax1.set_ylabel('Loss')
    ax1.tick_params('y')
    ax1.legend(loc='upper right', shadow=False)
    ax1.set_title('Model Loss', fontweight='bold')
                  
    # plot model accuracy
    ax2.plot(range(1, len(train_accuracy) + 1), train_accuracy, color='k', label='training')
    ax2.plot(range(1, len(train_val_accuracy) + 1), train_val_accuracy, color='k', ls='--', label='validation')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.tick_params('y')
    ax2.legend(loc='lower right', shadow=False)
    ax2.set_title('Model Accuracy', fontweight='bold')
                               

"""
# =============================================================================
# compare results
# =============================================================================
"""
def compare_results(sents_tagged, sents_pred, tags=[]):
    """
    Function to compare predicted results with actual results.
    
    Given an tagged sample and the predicted results, provide a cross-tab of 
    the actual v tagged results, for the requested tags.
    
    >>> sent = [[('σχεδιο', 'NO'), \
                 ('χαρτη', 'NO'), \
                 ('θεμελιωδων', 'NO'), \
                 ('δικαιωματων', 'NO'), \
                 ('της', 'AT'), \
                 ('ευρωπαϊκης', 'AJ'), \
                 ('ενωσης', 'NO')], \
                [('βρυξέλλες', 'NO'), \
                 (',', 'PU'), \
                 ('11', 'DI'), \
                 ('οκτωβρίoυ', 'NO'), \
                 ('2000', 'DI'), \
                 ('(', 'PU'), \
                 ('19.10', 'DI'), \
                 (')', 'PU')]]
    >>> pred = [[('σχεδιο', 'AJ'), \
                 ('χαρτη', 'AJ'), \
                 ('θεμελιωδων', 'NO'), \
                 ('δικαιωματων', 'NO'), \
                 ('της', 'AT'), \
                 ('ευρωπαϊκης', 'AJ'), \
                 ('ενωσης', 'NO')], \
                [('βρυξέλλες', 'NO'), \
                 (',', 'PU'), \
                 ('11', 'DI'), \
                 ('οκτωβρίoυ', 'NO'), \
                 ('2000', 'DI'), \
                 ('(', 'PU'), \
                 ('19.10', 'DI'), \
                 (')', 'PU')]]
    >>> a, b = compare_results(sent, pred)
    >>> list(a['NO'])
    [0, 0, 0, 5, 0]
    
    :param sents_tagged: A test corpus of tagged sentences.
    :type  sents_tagged: List of lists.
    :param  pred_tagged: A test corpus of sentences with predicted tags.
    :type   pred_tagged: List of lists.
    :param         tags: List of required tags.
    :type          tags: List
    :default       tags: [] - includes all tags.
    :rtype             : Panda dataframe and a list.
    """
    
    # flatten results
    sents_pred = [word for sent in sents_pred for word in sent]
    sents_tagged = [word for sent in sents_tagged for word in sent]
    
    # combine values and create dataframe of required tags
    ct_list = list(zip(sents_tagged, sents_pred))
    if tags != []:
        ct_list = [(word1, word2) if (word1[1] in tags and word2[1] in tags)
                   else (word1, (word2[0], 'XX')) if word1[1] in tags
                   else ((word1[0], 'XX'), word2) if word2[1] in tags
                   else ((word1[0], 'XX'), (word2[0], 'XX')) for (word1, word2) in ct_list]
    ct_comparison = [(tag_a, tag_p) for ((word_a, tag_a),(word_p, tag_p))
                     in ct_list]
    ct_comparison_df = pd.DataFrame(ct_comparison)
    ct_comparison_df.columns = ['predicted','actual']
    
    # perform crosstab and return result
    ct = pd.crosstab(ct_comparison_df['predicted'], ct_comparison_df['actual'])
    return(ct, ct_list)
    

"""
# =============================================================================
# compare results
# =============================================================================
"""
def compute_metrics(sents_tagged, sents_pred, tags=[]):
    """
    Function to compute the four individual metrics as well as F-Score.
    
    Given an tagged sample and the predicted results, provide precision,
    recall, sensitivity and specificity along with the F-Score. Only the micro
    F-Score is given and this is equal to the accuracy.
    
    >>> sent = [[('σχεδιο', 'NO'), \
                 ('χαρτη', 'NO'), \
                 ('θεμελιωδων', 'NO'), \
                 ('δικαιωματων', 'NO'), \
                 ('της', 'AT'), \
                 ('ευρωπαϊκης', 'AJ'), \
                 ('ενωσης', 'NO')], \
                [('βρυξέλλες', 'NO'), \
                 (',', 'PU'), \
                 ('11', 'DI'), \
                 ('οκτωβρίoυ', 'NO'), \
                 ('2000', 'DI'), \
                 ('(', 'PU'), \
                 ('19.10', 'DI'), \
                 (')', 'PU')]]
    >>> pred = [[('σχεδιο', 'AJ'), \
                 ('χαρτη', 'AJ'), \
                 ('θεμελιωδων', 'NO'), \
                 ('δικαιωματων', 'NO'), \
                 ('της', 'AT'), \
                 ('ευρωπαϊκης', 'AJ'), \
                 ('ενωσης', 'NO')], \
                [('βρυξέλλες', 'NO'), \
                 (',', 'PU'), \
                 ('11', 'DI'), \
                 ('οκτωβρίoυ', 'NO'), \
                 ('2000', 'DI'), \
                 ('(', 'PU'), \
                 ('19.10', 'DI'), \
                 (')', 'PU')]]
    >>> a, b = compute_metrics(sent, pred)
    >>> b
    0.8666666666666667
    
    :param sents_tagged: A test corpus of tagged sentences.
    :type  sents_tagged: List of lists.
    :param   sents_pred: A test corpus of sentences with predicted tags.
    :type    sents_pred: List of lists.
    :param         tags: List of required tags.
    :type          tags: List
    :default       tags: [] - includes all tags.
    :rtype             : Panda dataframe and a real number.
    """
    
    # flatten results
    sents_pred = [word for sent in sents_pred for word in sent]
    sents_tagged = [word for sent in sents_tagged for word in sent]
    
    # combine values and create dataframe of required tags
    ct_list = list(zip(sents_tagged, sents_pred))
    if tags != []:
        ct_list = [(word1, word2) if (word1[1] in tags and word2[1] in tags)
                   else (word1, (word2[0], 'XX')) if word1[1] in tags
                   else ((word1[0], 'XX'), word2) if word2[1] in tags
                   else ((word1[0], 'XX'), (word2[0], 'XX')) for (word1, word2) in ct_list]
    ct_comparison = [(tag_a, tag_p) for ((word_a, tag_a),(word_p, tag_p))
                     in ct_list]
    
    # get list of tags
    tags_list = set([x[0] for x in ct_comparison])
    tags_list = list(tags_list.union([x[1] for x in ct_comparison]))
    
    # for each tag compute details
    total = len(ct_comparison)
    metric_df = pd.DataFrame(columns=('count', 'precision', 'recall', 
                                      'sensitivity', 'specificity'))
    # running totals for micro (and macro F-Score - commented)
    TP = 0; FP = 0; FN = 0
#    prec = 0; rec = 0
    for tag in tags_list:
        tag_act = len([x for x in ct_comparison if x[0]==tag])
        tag_pred = len([x for x in ct_comparison if x[1]==tag])
        tag_true = len([x for x in ct_comparison if (x[0]==tag and x[1]==tag)])
        precision = tag_true/tag_pred if tag_pred !=0 else None
        recall = tag_true/tag_act if tag_act !=0 else None
        sensitivity = recall
        specificity = (total-tag_pred-tag_act+tag_true)/(total-tag_act)
        # update totals
        TP = TP + tag_true
        FP = FP + tag_pred - tag_true
        FN = FN + tag_act - tag_true
#        prec = prec + precision
#        rec = rec + recall
        # update dataframe
        metric_new = pd.DataFrame([[tag_act, precision, recall,
                                    sensitivity, specificity]],
                                  columns=('count', 'precision', 'recall', 
                                           'sensitivity', 'specificity'))
        metric_new.rename(index={0:tag}, inplace=True)
        metric_df = metric_df.append(metric_new)
        metric_df = metric_df.sort_values(by=['count'], ascending=False)

    # calculate F-Score
    f_score_micro = 2*(TP/(TP+FP))*(TP/(TP+FN))/((TP/(TP+FP))+(TP/(TP+FN)))
#    f_score_macro = 2*(prec/len(tags_list))*(rec/len(tags_list))/ \
#                      ((prec/len(tags_list))+(rec/len(tags_list)))        
    
    # return result
    return(metric_df, f_score_micro)
    

"""
# =============================================================================
# compare results
# =============================================================================
"""
def compute_sent_acc(sents_tagged, sents_pred):
    """
    Function to compute the setence level accuracy for a tagged corpus.
    
    Given an tagged sample and the predicted results, provide calculate the
    percentage of sentences which are correctly tagged.
    
    >>> sent = [[('σχεδιο', 'NO'), \
                 ('χαρτη', 'NO'), \
                 ('θεμελιωδων', 'NO'), \
                 ('δικαιωματων', 'NO'), \
                 ('της', 'AT'), \
                 ('ευρωπαϊκης', 'AJ'), \
                 ('ενωσης', 'NO')], \
                [('βρυξέλλες', 'NO'), \
                 (',', 'PU'), \
                 ('11', 'DI'), \
                 ('οκτωβρίoυ', 'NO'), \
                 ('2000', 'DI'), \
                 ('(', 'PU'), \
                 ('19.10', 'DI'), \
                 (')', 'PU')]]
    >>> pred = [[('σχεδιο', 'AJ'), \
                 ('χαρτη', 'AJ'), \
                 ('θεμελιωδων', 'NO'), \
                 ('δικαιωματων', 'NO'), \
                 ('της', 'AT'), \
                 ('ευρωπαϊκης', 'AJ'), \
                 ('ενωσης', 'NO')], \
                [('βρυξέλλες', 'NO'), \
                 (',', 'PU'), \
                 ('11', 'DI'), \
                 ('οκτωβρίoυ', 'NO'), \
                 ('2000', 'DI'), \
                 ('(', 'PU'), \
                 ('19.10', 'DI'), \
                 (')', 'PU')]]
    >>> compute_sent_acc(sent, pred)
    0.5
    
    :param sents_tagged: A test corpus of tagged sentences.
    :type  sents_tagged: List of lists.
    :param   sents_pred: A test corpus of sentences with predicted tags.
    :type    sents_pred: List of lists.
    :rtype             : Panda dataframe and a real number.
    """
    
    diffs = [[1-(sents_tagged[s][w][1]==sents_pred[s][w][1]) 
             for w in range(len(sents_tagged[s]))] 
             for s in range(len(sents_tagged))]
    sent_acc = len([s for s in diffs if sum(s)==0])/len(diffs)
    return(sent_acc)
    

"""
# =============================================================================
# deep learning tagger
# =============================================================================
"""       
def deep_learning_tag(sents, model, location):
    """
    Function to apply a Keras deep learning tagger.
    
    Given an tagged corpus and a trained Keras deep learning tagging model,
    return the tagged results.
    
    >>> print('No demo provided due to duration of runs...')
    No demo provided due to duration of runs...
    
    :param    sents: A corpus of sentences.
    :type     sents: List of lists.
    :param    model: A model name to be used
    :type     model: String.
    :param location: Directory containing the model details
    :type  location: String.
    :rtype         : List of lists.
    """
    
    """ load required model and details """
    pos_tag = load_model(location+model+'.h5')
    dict_vectorizer_train = load_tagger(location+model+'_DictVectorizer.pkl')
    label_encoder_train = load_tagger(location+model+'_LabelEncoder.pkl')   
    
    """ create the required input data """
    sents_tagged = [[(w, '') for w in s] for s in sents]
    X, _ = create_observation(sents_tagged)
    X = dict_vectorizer_train.transform(X)
    
    """ score and return results """
    # tag the text
    out = pos_tag.predict_classes(X)
    tag = [label_encoder_train.classes_[i] for i in out]
    # recreate the sentences
    tag_no = 0
    tag_res = []
    for sent in sents:
        tag_sent = []
        for word in sent:
            tag_sent.append((word, tag[tag_no]))
            tag_no = tag_no + 1
        tag_res.append(tag_sent)
    # return the results
    return(tag_res)


"""
# =============================================================================
# main
# =============================================================================
"""

if __name__ == '__main__':
    import doctest
    doctest.testmod()