"""
investigate_tagger_finetags

Scott Harrison - 2018-08-09

Code to investigate Modern Greek POS tagging of complex tags using NLTK.

This process looks at the various tagging options available and applies them 
to tagging Modern Greek text using the INTERA corpus with complex tags.

Note, as the testing done at this stage if to help refine the models, all test
estimates are derived from val_sents, leaving test_sents for the final 
performance evaluation.

The best tagger is stored as a pickle file for later evaluation.

"""

import numpy as np
from greek_nlp_corpus import read_corpus
from greek_nlp_pos import (tic, toc, untag, display_training_metrics, 
                           compute_metrics, add_features, compare_results,
                           create_observation, plot_model_performance,
                           save_tagger, deep_learning_tag)
from nltk.tag import (DefaultTagger, UnigramTagger, BigramTagger, 
                      TrigramTagger, AffixTagger, ClassifierBasedTagger)
from tag_util import train_brill_tagger
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.wrappers.scikit_learn import KerasClassifier

# set corpus and resource locations
CORPUS_DIR = './data/corpus/'
RESOURCES_DIR = './data/resources/'
RESULTS_DIR = './data/results/'

# set user constants
TAG_LENGTH = 20
PROPORTION = 100


""" 0. set up the corpora for training and testing of tagging methods """
"""    create a more complex version of the tags, incorporating case """

# load the corpus as tagged sentences
train_sents, val_sents, test_sents = read_corpus('INTERA', 
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
train_sents = update_tags(train_sents)
val_sents = update_tags(val_sents)
test_sents = update_tags(test_sents)


"""
# =============================================================================
# finalise a sequential tagger
# =============================================================================
"""

""" 1. run tagger with different corpus size (50% and 100%) """
# backoff tagger
tag1_eval = dict()
# train with backoff and Brill
tic(); 
tag1_tagger = DefaultTagger('NO')
tag1_tagger = AffixTagger(train_sents, affix_length=-1, backoff=tag1_tagger)
tag1_tagger = AffixTagger(train_sents, affix_length=-2, backoff=tag1_tagger)
tag1_tagger = AffixTagger(train_sents, affix_length=-3, backoff=tag1_tagger)
tag1_tagger = AffixTagger(train_sents, affix_length=-4, backoff=tag1_tagger)
tag1_tagger = AffixTagger(train_sents, affix_length=-5, backoff=tag1_tagger)
tag1_tagger = UnigramTagger(train_sents, cutoff=3, backoff=tag1_tagger)
tag1_tagger = BigramTagger(train_sents, backoff=tag1_tagger)
tag1_tagger = TrigramTagger(train_sents, backoff=tag1_tagger)
tag1b_tagger = train_brill_tagger(tag1_tagger, train_sents, True,
                                         max_rules=100)
tag1_eval['train_time'] = toc()
# test
tic(); tag1_eval['test_accuracy'] = tag1b_tagger.evaluate(val_sents)
tag1_eval['test_time'] = toc()
# display results
display_training_metrics(tag1_eval)


"""
# =============================================================================
# finalise a classification-based tagger
# =============================================================================
"""

""" 1. Naive Bayes classifier tagger with features and Brill """
nb_eval = dict()
# train
tic(); 
nb_tagger = ClassifierBasedTagger(train=train_sents, 
                                    feature_detector=add_features)
nb_eval['train_time'] = toc()
# test
tic(); nb_eval['test_accuracy'] = nb_tagger.evaluate(val_sents)
nb_eval['test_time'] = toc()
# display results
display_training_metrics(nb_eval)


"""
# =============================================================================
# finalise a deep learning tagger
# =============================================================================
"""

""" 1. prepare the data """
# for train, test and validation
train_X, train_y = create_observation(train_sents)
val_X, val_y = create_observation(val_sents)

# convert features to vectors
dict_vectorizer = DictVectorizer(sparse=True)
dict_vectorizer.fit(train_X)
train_X = dict_vectorizer.transform(train_X)
val_X = dict_vectorizer.transform(val_X)

# convert outputs to vectors
# first encode as integers
label_encoder = LabelEncoder()
label_encoder.fit(train_y)
train_y = label_encoder.transform(train_y)
val_y = label_encoder.transform(val_y)
# convert to dummy variables
train_y = np_utils.to_categorical(train_y)
val_y = np_utils.to_categorical(val_y)
# pad vectors where necesary
train_add = np.zeros(len(label_encoder.classes_) - len(train_y[0]))
train_y = np.array([np.append(v,train_add) for v in train_y])
val_add = np.zeros(len(label_encoder.classes_) - len(val_y[0]))
val_y = np.array([np.append(v,val_add) for v in val_y])


""" 2. create and train the Keras model """

# build the model
tic()
# build the model
tic()
def build_model(input_dim, hidden_neurons, output_dim):
    # construct, compile and return a Keras model
    model = Sequential([Dense(64, input_dim=input_dim),
                        Activation('relu'),
                        Dropout(0.2),
                        Dense(hidden_neurons),
                        Activation('relu'),
                        Dropout(0.2),
                        Dense(output_dim, activation='softmax')])
    model.compile(loss='categorical_crossentropy', 
                  optimizer='adam', 
                  metrics=['accuracy'])
    return(model)
    
# set model parameters and define the classifier
model_params = {
    'build_fn': build_model,
    'input_dim': train_X.shape[1],
    'hidden_neurons': 32,
    'output_dim': train_y.shape[1],
    'epochs': 5,
    'batch_size': 256,
    'verbose': 1,
    'validation_data': (val_X, val_y),
    'shuffle': True
}
pos_model = KerasClassifier(**model_params)

# train the model
pos_model_history = pos_model.fit(train_X, train_y)
deeplearn_eval = dict()
deeplearn_eval['train_time'] = toc()
print(deeplearn_eval['train_time'])

# review training results
plot_model_performance(pos_model_history)


""" 3. convert the model to NLTK for testing """
""" 4. save and test the model """

# temporarily save the trained model, history and details
pos_model.model.save(RESOURCES_DIR+'Greek_POS_ctag_DL.h5')
save_tagger(RESULTS_DIR+'Greek_POS_ctag_DL_History.pkl', 
            pos_model_history.history)   
save_tagger(RESOURCES_DIR+'Greek_POS_ctag_DL_DictVectorizer.pkl', 
            dict_vectorizer)   
save_tagger(RESOURCES_DIR+'Greek_POS_ctag_DL_LabelEncoder.pkl', 
            label_encoder)   

# test the results
tic()
val_tagged = deep_learning_tag(untag(val_sents), 'Greek_POS_ctag_DL', 
                               RESOURCES_DIR)
deeplearn_eval['test_time'] = toc()
_, deeplearn_eval['test_accuracy'] = compute_metrics(val_sents, val_tagged)
display_training_metrics(deeplearn_eval)


"""
# =============================================================================
# finalise the details
# =============================================================================
"""

""" 1. look at errors and metrics """
metrics, _ = compute_metrics(val_sents, val_tagged)
val_sents_2 = [[(w,t[:2]) for (w,t) in sent] for sent in val_sents]
val_tagged_2 = [[(w,t[:2]) for (w,t) in sent] for sent in val_tagged]
metrics_2, acc_2 = compute_metrics(val_sents_2, val_tagged_2)
output, comparison = compare_results(val_sents, val_tagged, tags=['NOAC', 'NONM',
                                                                  'AJAC', 'AJNM'])

""" 2. save the taggers """
save_tagger(RESOURCES_DIR+'Greek_POS_ctag_seq.pkl', tag1b_tagger)
save_tagger(RESOURCES_DIR+'Greek_POS_ctag_class.pkl', nb_tagger)
