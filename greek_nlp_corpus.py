"""
greek_nlp_corpus

Scott Harrison - 2018-07-28

Functions to read in various Greek corpora

This module reads in various Greek corpora for either training or testing
purposes and also allows a fraction on the corpus to be read.

"""

# module attributes - for inline examples
DEMO_DIR = './data/_demo_/'                        # location of demo resources
CORP_DIR = './data/corpus/'                           # location of the corpora

import os
import random
from nltk.corpus import TaggedCorpusReader


"""
# =============================================================================
# main corpus function
# =============================================================================
"""
def read_corpus(corpus, role='test', proportion=100, tag_length=2):
    """
    Sets up and completes the reading in of the requested corpus.
    
    Given a corpus along with an optional role and proportion of the corpus,
    reads in the corpus and outputs the required sub-corpora (train, val, test 
    or test only).
    
    :param        corpus: The name of the corpus.
    :type         corpus: String (one of {'INTERA', 'UDGreek', 'tagged_texts'})
    :param          role: Whether the corpus is for training or testing only.
    :type           role: String (one of {'train', 'test'})
    :default        role: 'test'
    :param    proportion: Proportion of the corpus to include (INTERA only)
    :type     proportion: Integer
    :default  proportion: 100
    :param    tag_length: Length of tag to include (only suitable for INTERA)
    :type     tag_length: Integer
    :default  tag_length: 2
    :rtype              : One (test) or three (train) lists of sentences.
    :raise exception    : If an invalid corpus provided.
    """    

    # constants
    TEST_PROP = 0.2
    SHUFFLE_SEED = 2101
    
    # get the required files
    try:
        corp_files = os.listdir(CORP_DIR+corpus)
    except:
        # warn about invalid corpus
        return(print('ERR - read_corpus: ensure the corpus name is correct. '
                     'One of ''INTERA'', ''UDGreek'', ''tagged_texts'''))
        
    # select required proportion (INTERA only)
    if corpus == 'INTERA' and proportion < 100:
        sample_size = int(proportion/100*len(corp_files))
        random.Random(SHUFFLE_SEED).shuffle(corp_files)
        corp_files = corp_files[:sample_size]
        
    # read in test files (includes tagged_texts)
    if corpus == 'tagged_texts' or role == 'test':               # testing only
        print('\n=== Test ===')
        test_sents = read_sub_corpus(corpus, corp_files, tag_length=tag_length)
        return([], [], test_sents)
        
    # read in training files
    if role == 'train':
        
        # UD Greek
        if corpus == 'UDGreek':
            # load subsets
            print('\n=== Train ===')
            train_sents = read_sub_corpus(corpus, ['udg_train.txt'], 
                                          tag_length=tag_length)
            print('\n=== Validation ===')
            val_sents = read_sub_corpus(corpus, ['udg_val.txt'], 
                                        tag_length=tag_length)
            print('\n=== Test ===')
            test_sents = read_sub_corpus(corpus, ['udg_test.txt'], 
                                         tag_length=tag_length)
            
        # INTERA
        if corpus == 'INTERA':
            # create sub sets 
            sample_size = int(TEST_PROP*len(corp_files))
            random.Random(SHUFFLE_SEED+1).shuffle(corp_files)
            test_files = corp_files[:sample_size]
            val_files = corp_files[sample_size:][:sample_size]
            train_files = corp_files[sample_size:][sample_size:]
            # load subsets
            print('\n=== Train ===')
            train_sents = read_sub_corpus(corpus, train_files, 
                                          tag_length=tag_length)
            print('\n=== Validation ===')
            val_sents = read_sub_corpus(corpus, val_files, 
                                        tag_length=tag_length)
            print('\n=== Test ===')
            test_sents = read_sub_corpus(corpus, test_files, 
                                         tag_length=tag_length)
            
        # return train, val and test
        return(train_sents, val_sents, test_sents)
    

"""
# =============================================================================
# corpus read function
# =============================================================================
"""
def read_sub_corpus(corpus, files_req, tag_length=2):
    """
    Read in the requested files from the requested corpus.
    
    Given a corpus and filenames, reads in and cleans the pos tagged data,
    including truncating tags for INTERA.
    
    :param        corpus: The name of the corpus.
    :type         corpus: String (one of {'INTERA', 'UDGreek', 'tagged_texts'})
    :param     files_req: The files to be read.
    :type      files_req: List
    :param    tag_length: Length of tag to include (INTERA only)
    :type     tag_length: Integer
    :default  tag_length: 2
    :rtype              : One (test) or three (train) lists of sentences.
    :raise exception    : If an invalid corpus provided.
    """
    
    # load the corpus as tagged sentences
    corp_sents = list()
    # for each file
    for file_name in files_req:
        # mask parenthesis
        file_name = file_name.replace('(','\(').replace(')', '\)')
        corp_raw = TaggedCorpusReader(CORP_DIR+corpus, file_name)
        corp_sents.extend(corp_raw.tagged_sents())
        
    print('Files read    : '+str(len(files_req)))
    print('Sentences read: '+ str(len(corp_sents)))
    print('Words read    : '+str(sum([len(x) for x in corp_sents])))
    
    # clean the tags - replace missing with '' - and simplify
    corp_sents = [list(map(lambda x: (x[0],'') if x[1]==None else x, sent)) 
                  for sent in corp_sents]
    corp_sents = [[(word,tag[:tag_length]) for (word,tag) in sent] 
                  for sent in corp_sents]
    
    # return the loaded files
    return(corp_sents)
        
    
# =============================================================================
# #EXAMPLES
# # tagged_texts
# _, _, tz = read_corpus('tagged_texts')
# # UD Greek
# ux, uy, uz = read_corpus('UDGreek', role='train', proportion=50)
# # INTERA
# ix, iy, iz = read_corpus('INTERA', proportion=10, tag_length=4, role='train')
# =============================================================================
