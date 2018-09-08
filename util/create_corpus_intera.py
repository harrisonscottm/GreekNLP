"""
# =============================================================================
# create_corpus_intera
# 
# Scott Harrison - 2018-07-06
# 
# Creates a tagged Modern Greek corpus from the INTERA corpus.
#
# One-off code to convert xml corpus files to POS for use with NLTK
# TaggedCorpusReader.
# 
# =============================================================================
"""

import xml.etree.ElementTree as et
import os

# set working folders and output folders
ROOT = <root directory>
CORPUS = ROOT+<corpus directory>
SUBJECTS = ['EDU', 'ENV', 'HEALTH', 'LAW', 'TOUR']
OUT_POS = CORPUS+'Corpus_INTERA_POS_Case/'

""" define POS function """
def create_pos(file_in, file_out):
    # open file to write results
    file_write = open(file_out, 'w', encoding='utf-8')
    # extract values
    for child in et.parse(file_in).getroot().iter('s'):
        # id = child.attrib.get('id', None)                # not used currently
        text_tagged = ''
        for item in child.iter('tok'):
            orth = item.find('orth').text
            ctag = item.find('ctag').text
            # token = item.attrib.get('id', None)          # not used currently
            # base = item.find('base').text                # not used currently
            # msd = item.find('msd').text                  # not used currently
            # create and append tagged text
            text_tagged = ' '.join((text_tagged, orth+'/'+ctag.upper()))
        # write results
        file_write.write(str.strip(text_tagged)+'\n\n') 
    # close file
    file_write.close() 

""" loop through all folders and files for POS corpus """
# for each subject
for subject in SUBJECTS:
    # get all files for POS
    dir_pos = CORPUS+'EL_'+subject+'_xcesAna/'
    files_pos = os.listdir(dir_pos)
    # loop through all POS files and create corpus
    for file_pos in files_pos:
        file_in_curr = dir_pos+file_pos
        file_out_curr = OUT_POS+file_pos[:-15]+'.txt'
        create_pos(file_in_curr, file_out_curr)