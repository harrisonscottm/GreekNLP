"""
investigate_tagger_NLTK_sequential

Scott Harrison - 2018-07-14

Code to investigate Modern Greek POS tagging using NLTK.

This process looks at the various sequential tagging options available in
NLTK and applies them to tagging Modern Greek text using the INTERA corpus.

Note, as the testing done at this stage if to help refine the models, all test
estimates are derived from val_sents, leaving test_sents for the final 
performance evaluation.

The best tagger is stored as a pickle file for later evaluation.

"""

from greek_nlp_corpus import read_corpus
from greek_nlp_pos import (tic, toc, untag, display_training_metrics, 
                           compute_metrics, compare_results,
                           find_pos_most_common, create_regexp_list,
                           save_tagger, plot_two_axis)
from nltk.tag import (DefaultTagger, UnigramTagger, BigramTagger, 
                      TrigramTagger, AffixTagger, RegexpTagger)
from tag_util import train_brill_tagger

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
# investigate individual NLTK sequential tagging components
# =============================================================================
"""

""" 0. create function to run and test components """
def test_tagger(tagger_name, tagger_input, test_data, **kwargs):
    # initialise results
    tagger_eval = dict()
    # train
    tic(); tagger_tagger = tagger_name(tagger_input, **kwargs)
    tagger_eval['train_time'] = toc()
    # test
    tic(); tagger_eval['test_accuracy'] = tagger_tagger.evaluate(test_data)
    tagger_eval['test_time'] = toc()
    # show results
    display_training_metrics(tagger_eval)
    
""" 1. default tagger with most common tag """
most_common_tag = find_pos_most_common(train_sents, item_type='tag', item_n=1)
test_tagger(DefaultTagger, most_common_tag[0][0], val_sents)


""" 2. unigram tagger """
test_tagger(UnigramTagger, train_sents, val_sents, cutoff=0)


""" 3. bigram tagger """
test_tagger(BigramTagger, train_sents, val_sents)


""" 4. trigram tagger """
test_tagger(TrigramTagger, train_sents, val_sents)


""" 5. create regexp tagger """
regexp_list = create_regexp_list('Open_Word_Patterns.xlsx', RESOURCES_DIR)
test_tagger(RegexpTagger, regexp_list, val_sents)


""" 6. affix tagger """
test_tagger(AffixTagger, train_sents, val_sents, affix_length=-3)


"""
# =============================================================================
# compound taggers using sequential taggers and backoff
# =============================================================================
"""

""" 1. create a tagger utilising: 
       n-gram, unigram, regexp and default taggers """
tag2_eval = dict()
# train with backoff
tic(); tag2_input = create_regexp_list('Open_Word_Patterns.xlsx',RESOURCES_DIR)
tag2_tagger = DefaultTagger('NO')
tag2_tagger = RegexpTagger(tag2_input, backoff=tag2_tagger)
tag2_tagger = UnigramTagger(train_sents, cutoff=3, backoff=tag2_tagger)
tag2_tagger = BigramTagger(train_sents, backoff=tag2_tagger)
tag2_tagger = TrigramTagger(train_sents, backoff=tag2_tagger)
tag2_eval['train_time'] = toc()
# test
tic(); tag2_eval['test_accuracy'] = tag2_tagger.evaluate(val_sents)
tag2_eval['test_time'] = toc()
# display results
display_training_metrics(tag2_eval)


""" 2. create a tagger utilising: 
       n-gram, unigram, affix and default taggers """
tag1_eval = dict()
# train with backoff
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
tag1_eval['train_time'] = toc()
# test
tic(); tag1_eval['test_accuracy'] = tag1_tagger.evaluate(val_sents)
tag1_eval['test_time'] = toc()
# display results
display_training_metrics(tag1_eval)


"""
# =============================================================================
# add Brill tagger to compound taggers
# =============================================================================
"""

""" 0. look at optimum number of rules """
tag1b_eval_opt = list()
for num in [10, 20, 50, 100, 200, 500]:
    print(num)
    temp = train_brill_tagger(tag1_tagger, train_sents, True, max_rules=num)
    act = len(temp.rules())
    tic()
    res = temp.evaluate(val_sents)
    dur = toc()
    tag1b_eval_opt.append((num, act, res, dur))
print(tag1b_eval_opt)
# plot results of different number of rules
plot_two_axis('Number of rules applied', [d[1] for d in tag1b_eval_opt],
              'Accuracy (%)', [d[2] for d in tag1b_eval_opt],
              'Scoring Time (s)', [d[3] for d in tag1b_eval_opt])


""" 1. add Brill tagger to affix compound tagger """
tag1b_eval = dict()
# train
tic(); tag1b_tagger = train_brill_tagger(tag1_tagger, train_sents, True,
                                         max_rules=100)
tag1b_eval['train_time'] = toc()
# test
tic(); tag1b_eval['test_accuracy'] = tag1b_tagger.evaluate(val_sents)
tag1b_eval['test_time'] = toc()
# display results
display_training_metrics(tag1b_eval)
tag1b_tagger.rules()


"""
# =============================================================================
# finalise a sequential tagger
# =============================================================================
"""

""" 0. look at optimum size of corpus - minus Brill tagger """
tag1b_eval_opt = list()
for corp_size in [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]:
    print(corp_size)
    # load data
    train_sents, val_sents, test_sents = read_corpus('INTERA', 
                                                 role='train',
                                                 proportion=corp_size,
                                                 tag_length=TAG_LENGTH)
    # create tagger
    tag1_tagger = DefaultTagger('NO')
    tag1_tagger = AffixTagger(train_sents, affix_length=-1, backoff=tag1_tagger)
    tag1_tagger = AffixTagger(train_sents, affix_length=-2, backoff=tag1_tagger)
    tag1_tagger = AffixTagger(train_sents, affix_length=-3, backoff=tag1_tagger)
    tag1_tagger = AffixTagger(train_sents, affix_length=-4, backoff=tag1_tagger)
    tag1_tagger = AffixTagger(train_sents, affix_length=-5, backoff=tag1_tagger)
    tag1_tagger = UnigramTagger(train_sents, cutoff=3, backoff=tag1_tagger)
    tag1_tagger = BigramTagger(train_sents, backoff=tag1_tagger)
    tag1_tagger = TrigramTagger(train_sents, backoff=tag1_tagger)
    tic()
    res = tag1_tagger.evaluate(val_sents)
    dur = toc()
    tag1b_eval_opt.append((corp_size, res, dur))
print(tag1b_eval_opt)
# plot results of different corpus sizes
plot_two_axis('Proportion of corpus utilised', [d[0] for d in tag1b_eval_opt],
              'Accuracy (%)', [d[1] for d in tag1b_eval_opt],
              'Scoring Time (s)', [d[2] for d in tag1b_eval_opt])


""" 1. re-run tagger with reduced corpus - with Brill tagger """
train_sents, val_sents, test_sents = read_corpus('INTERA', 
                                                 role='train',
                                                 proportion=70,
                                                 tag_length=TAG_LENGTH)
tag1b_eval = dict()
# train
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
tag1b_eval['train_time'] = toc()
# test
tic(); tag1b_eval['test_accuracy'] = tag1b_tagger.evaluate(val_sents)
tag1b_eval['test_time'] = toc()
# display results
display_training_metrics(tag1b_eval)


""" 2. look at errors and metrics """
tag1b_res = tag1b_tagger.tag_sents(untag(val_sents))
metrics, _ = compute_metrics(val_sents, tag1b_res)
output, comparison = compare_results(val_sents, tag1b_res, 
                                     tags=['NO', 'AJ'])


""" 3. save the tagger """
save_tagger(RESOURCES_DIR+'Greek_POS_seq.pkl',tag1b_tagger)