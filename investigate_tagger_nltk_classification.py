"""
investigate_tagger_NLTK_classification

Scott Harrison - 2018-07-24

Code to investigate Modern Greek POS tagging using NLTK classification.

This process looks at the various classification tagging options available in
NLTK and applies them to tagging Modern Greek text using the INTERA corpus.

Note, as the testing done at this stage if to help refine the models, all test
estimates are derived from val_sents, leaving test_sents for the final 
performance evaluation.

The best tagger is stored as a pickle file for later evaluation.

"""

from greek_nlp_corpus import read_corpus
from greek_nlp_pos import (tic, toc, display_training_metrics, add_features,
                           untag, compute_metrics, compare_results, 
                           save_tagger, plot_two_axis)
from nltk.tag import tnt
from nltk.tag.sequential import ClassifierBasedTagger, ClassifierBasedPOSTagger

# set corpus and resource locations
CORPUS_DIR = './data/corpus/'
RESOURCES_DIR = './data/resources/'

# set user constants
TAG_LENGTH = 2
PROPORTION = 100


""" 0. set up the corpora for training and testing of tagging methods """

# load the corpus as tagged sentences
train_sents, val_sents, test_sents = read_corpus('INTERA', 
                                                 role='train',
                                                 proportion=PROPORTION,
                                                 tag_length=TAG_LENGTH)


"""
# =============================================================================
# investigate NLTK classification tagging options
# =============================================================================
"""

""" 1. TNT tagger """
tnt_eval = dict()
# train
tic(); tnt_tagger = tnt.TnT()
tnt_tagger.train(train_sents)
tnt_eval['train_time'] = toc()
# test
tic(); tnt_eval['test_accuracy'] = tnt_tagger.evaluate(val_sents)
tnt_eval['test_time'] = toc()
# display results
display_training_metrics(tnt_eval)


""" 2. Naive Bayes classifier tagger """
nb_eval = dict()
# train
tic()
nb_tagger = ClassifierBasedPOSTagger(train=train_sents)
nb_eval['train_time'] = toc()
# test
tic(); nb_eval['test_accuracy'] = nb_tagger.evaluate(val_sents)
nb_eval['test_time'] = toc()
# display results
display_training_metrics(nb_eval)


""" 3. Naive Bayes classifier tagger with features """
nb_eval = dict()
# train
tic(); nb_tagger = ClassifierBasedTagger(train=train_sents, 
                                         feature_detector=add_features)
nb_eval['train_time'] = toc()
# test
tic(); nb_eval['test_accuracy'] = nb_tagger.evaluate(val_sents)
nb_eval['test_time'] = toc()
# display results
display_training_metrics(nb_eval)


"""
# =============================================================================
# finalise a model tagger
# =============================================================================
"""

""" 0. look at optimum size of corpus """
tag3_eval_opt = list()
for corp_size in [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]:
    print(corp_size)
    # load data
    train_sents, val_sents, test_sents = read_corpus('INTERA', 
                                                 role='train',
                                                 proportion=corp_size,
                                                 tag_length=TAG_LENGTH)
    # create tagger
    tag3_tagger = ClassifierBasedTagger(train=train_sents, 
                                      feature_detector=add_features)
    tic()
    res = tag3_tagger.evaluate(val_sents)
    dur = toc()
    tag3_eval_opt.append((corp_size, res, dur))
print(tag3_eval_opt)
# plot results of different corpus sizes
plot_two_axis('Proportion of corpus utilised', [d[0] for d in tag3_eval_opt],
              'Accuracy (%)', [d[1] for d in tag3_eval_opt],
              'Scoring Time (s)', [d[2] for d in tag3_eval_opt])


""" 1. re-run tagger with changed corpus size """

# load the corpus as tagged sentences
train_sents, val_sents, test_sents = read_corpus('INTERA', 
                                                 role='train',
                                                 proportion=70,
                                                 tag_length=TAG_LENGTH)

# Naive Bayes tagger
tag3_eval = dict()
# train
tic()
tag3_tagger = ClassifierBasedTagger(train=train_sents, 
                                    feature_detector=add_features)
tag3_eval['train_time'] = toc()
# test
tic(); tag3_eval['test_accuracy'] = tag3_tagger.evaluate(val_sents)
tag3_eval['test_time'] = toc()
# display results
display_training_metrics(tag3_eval)


""" 2. look at errors and metrics """
tag3_res = tag3_tagger.tag_sents(untag(val_sents))
metrics, _ = compute_metrics(val_sents, tag3_res)
output, comparison = compare_results(val_sents, tag3_res, 
                                     tags=['NO', 'AJ', 'AD', 'PN'])


""" 3. save the tagger """
save_tagger(RESOURCES_DIR+'Greek_POS_class.pkl', tag3_tagger)