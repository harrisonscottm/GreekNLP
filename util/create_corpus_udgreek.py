"""
# =============================================================================
# create_corpus_udgreek
# 
# Scott Harrison - 2018-07-28
# 
# Creates a tagged Modern Greek corpus from the UD Greek corpus.
#
# One-off code to convert conllu text corpus files to POS for use with NLTK
# TaggedCorpusReader.
# 
# =============================================================================
"""

import os
import pandas as pd

# set working folders and output folders
ROOT = <root directory>
CORPUS = ROOT+<corpus directory>
IN_POS = CORPUS+'Raw/'
OUT_POS = CORPUS+'UDGreek/'

"""
# =============================================================================
#  function to read in file, format and write out tagged corpus
# """
# =============================================================================

def read_write_udg(infile, outfile):

    """ read in file """
    corp_df = pd.read_csv(infile, sep='\t', comment='#', header=None)
                          
    """ select required columns - 0, 1, 3, 5 and 7 """
    corp_df_ref = corp_df[[0, 1,3,5,7]]
    corp_df_ref.columns = ['Ind', 'Word', 'Tag', 'Info', 'Alt']
    
    """ recode the tags """
    corp_df_ref.loc[corp_df_ref.Info=='Abbr=Yes','Tag'] = 'AB'
    corp_df_ref.loc[corp_df_ref.Tag=='NUM','Tag'] = 'DI'
    corp_df_ref.loc[corp_df_ref.Tag=='ADJ','Tag'] = 'AJ'
    corp_df_ref.loc[corp_df_ref.Tag=='ADP','Tag'] = 'AS'
    corp_df_ref.loc[corp_df_ref.Tag=='ADV','Tag'] = 'AD'
    corp_df_ref.loc[corp_df_ref.Tag=='AUX','Tag'] = 'VB'
    corp_df_ref.loc[corp_df_ref.Tag=='CCONJ','Tag'] = 'CJ'
    corp_df_ref.loc[corp_df_ref.Tag=='DET','Tag'] = 'AT'
    corp_df_ref.loc[corp_df_ref.Tag=='NOUN','Tag'] = 'NO'
    corp_df_ref.loc[corp_df_ref.Tag=='PART','Tag'] = 'PT'
    corp_df_ref.loc[corp_df_ref.Tag=='PRON','Tag'] = 'PN'
    corp_df_ref.loc[corp_df_ref.Tag=='PROPN','Tag'] = 'NO'
    corp_df_ref.loc[corp_df_ref.Tag=='PUNCT','Tag'] = 'PU'
    corp_df_ref.loc[corp_df_ref.Tag=='SCONJ','Tag'] = 'CJ'
    corp_df_ref.loc[corp_df_ref.Tag=='VERB','Tag'] = 'VB'
    corp_df_ref.loc[corp_df_ref.Tag=='SYM','Tag'] = 'AB'
    corp_df_ref.loc[corp_df_ref.Info=='Foreign=Yes', 'Tag'] = 'RG'
    corp_df_ref.loc[corp_df_ref.Word=='	', 'Word'] = '"'
    #set(corp_df_ref['Tag'])
    #set(corp_df_ref['Info'])
    
    """ remove separated compounds """
    for i in range(len(corp_df_ref)):
        if i > 0:
            if corp_df_ref.loc[i-1].Tag == '_':
                corp_df_ref.loc[i].Tag = 'DEL'
        if i > 1:
            if corp_df_ref.loc[i-2].Tag == '_':
                corp_df_ref.loc[i].Tag = 'DEL'
    corp_df_ref = corp_df_ref[corp_df_ref.Tag != 'DEL']
    
    """ provide correct tag for compounds """
    #set(corp_df_ref[corp_df_ref.Tag == '_'].Word)
    corp_df_ref.loc[corp_df_ref.Tag == '_', 'Tag'] = 'AS'
            
    """ print out all the tagged words """
    corp_df_ref = corp_df_ref.reset_index(drop=True)
    file_write = open(outfile, 'w', encoding='utf-8')
    text_tagged = ''
    for i in range(len(corp_df_ref)):
        if corp_df_ref.loc[i].Ind == '1' or corp_df_ref.loc[i].Ind == '1-2':
            if i > 0:
                file_write.write(str.strip(text_tagged)+'\n\n') 
            text_tagged = ''
        text_tagged = ' '.join((text_tagged, corp_df_ref.loc[i].Word+'/'+
                                             corp_df_ref.loc[i].Tag.upper()))
    file_write.write(str.strip(text_tagged)+'\n\n')
    file_write.close()


""" run for each file """
read_write_udg(IN_POS+'el_gdt-ud-train.conllu',OUT_POS+'udg_train.txt')
read_write_udg(IN_POS+'el_gdt-ud-dev.conllu',OUT_POS+'udg_val.txt')
read_write_udg(IN_POS+'el_gdt-ud-test.conllu',OUT_POS+'udg_test.txt')
