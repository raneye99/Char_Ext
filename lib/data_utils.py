#utility functions for data (ie loading)

#import libraries
import pandas as pd
import numpy as np
import os
import sys
import re

# add lib package to path if it is not there already

# try: 
#     src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
#     if src_path not in sys.path:
#         sys.path.extend([src_path])
# except NameError:
#     print("Issue with adding to path, check if __file__ is defined")

# datapath = src_path + '/Data/'
# PLpath = datapath + 'AnnotatedPLData/PLCoref'
# CENpath = datapath + 'AnnotatedCENData/CENCoref'
# ONpath = datapath + 'AnnotatedONData/ONCoref'

def read_story(path):
    '''
    function to read in text data

    input: path, the path to where our input data is located 

    input data is structured in the following format: int int coref
    where first int is a binary flag for animacy and the second int 
    is binary flag for character, coref refers to coreference chains
    
    output: pandas dataframe with storyid 
    '''
    #get title/corpus identifier from path
    title = re.search(r'(?s:.*)\/(.*?)\.txt', path).group(1)

    #read in corpus from path
    corp = pd.read_csv(path, sep='\t', header=None)
    corp.columns = ['animacy', 'character', 'coref_chain']
    
    #create new columns 
    corp["corpusID"] = title
    corp["coref_chain"] = corp["coref_chain"].str.split("|").str[1:-1]
    corp["chain_head"] = corp["coref_chain"].str[0]
    corp["head_of_head"]= corp["chain_head"].str.split(" ").str[-2]
    corp["chain_len"] = corp["coref_chain"].str.len()
    corp["CL"] = (corp["chain_len"]-np.mean(corp["chain_len"]))/np.std(corp["chain_len"])
   
    return corp
    

    

