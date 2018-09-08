"""
investigate_tagger_keras_deeplearning

Scott Harrison - 2018-07-22

Code to investigate Modern Greek POS tagging using Keras deep learning.

This process builds a deep learning model using Keras and applies it to 
tagging Modern Greek text using the INTERA corpus.

The process is based on the tutorial 'Part-of-Speech tagging tutorial with the 
Keras Deep Learning library' by Axel Bellec at becominghuman.ai, with
alterations to orientate the learning towards a Greek corpus.

Note, as the testing done at this stage if to help refine the models, all test
estimates are derived from val_sents, leaving test_sents for the final 
performance evaluation.

The best tagger is stored as a series of files for later evaluation.

"""

import numpy as np
from greek_nlp_corpus import read_corpus
from greek_nlp_pos import (tic, toc, find_pos_most_common, create_observation,
                           plot_model_performance, untag,
                           display_training_metrics, compute_metrics,
                           compare_results, save_tagger, deep_learning_tag)
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import plot_model

# set corpus and resource locations
CORPUS_DIR = './data/corpus/'
RESOURCES_DIR = './data/resources/'
RESULTS_DIR = './data/results/'

# set user constants
TAG_LENGTH = 2
PROPORTION = 100


""" 0. set up the corpora for training and testing of tagging methods """

# increment corpus size by 10 and load the corpus as tagged sentences
train_sents, val_sents, test_sents = read_corpus('INTERA', 
                                                 role='train',
                                                 proportion=PROPORTION,
                                                 tag_length=TAG_LENGTH)


""" 1. look at the number and values of tags (outputs) """

# review tags
train_tags = find_pos_most_common(train_sents, item_type='tag')
print('Tags ('+str(len(train_tags))+'): ', train_tags)


""" 2. create data (features and outputs) for deep learning """

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


""" 3. create and train the Keras model """

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
plot_model(pos_model.model, to_file=RESULTS_DIR+'Greek_POS_deep_model.png', 
           show_shapes=True)


""" 4. save and test the model """

# temporarily save the trained model, history and details
pos_model.model.save(RESOURCES_DIR+'Greek_POS_DL.h5')
save_tagger(RESULTS_DIR+'Greek_POS_DL_History.pkl', pos_model_history.history)   
save_tagger(RESOURCES_DIR+'Greek_POS_DL_DictVectorizer.pkl', dict_vectorizer)   
save_tagger(RESOURCES_DIR+'Greek_POS_DL_LabelEncoder.pkl', label_encoder)   

# test the results
tic()
val_tagged = deep_learning_tag(untag(val_sents), 'Greek_POS_DL', RESOURCES_DIR)
deeplearn_eval['test_time'] = toc()
_, deeplearn_eval['test_accuracy'] = compute_metrics(val_sents, val_tagged)
display_training_metrics(deeplearn_eval)


""" 5. look at errors and metrics """
metrics, _ = compute_metrics(val_sents, val_tagged)
output, comparison = compare_results(val_sents, val_tagged, tags=['NO', 'AJ', 'PN'])
