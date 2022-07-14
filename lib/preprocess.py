#functions for data preprocessing and feature extraction

#import libraries
import pandas as pd
import numpy as np
import os
import sys
import allennlp
import torch
import spacy
nlp = spacy.load('en_core_web_sm')

#demo use for allenlp
from allennlp.predictors.predictor import Predictor
import allennlp_models.tagging

def semantic_subj(storypath):
    '''
    input: path to story text
    output: list of semantic subject for story
    '''

    #load in raw texts
    f = open(storypath, 'r')
    doc = f.read()
    f.close()

    #remove newlines
    new = doc.replace('\n', ' ').replace("\'", '"')
    # print(new)

    #split doc into sentences using spacy 
    sentences = [i for i in nlp(new).sents]

    #convert back to text to feed into predictor
    sent_list = [i.text for i in sentences]
    sent_list = [i.strip() for i in sent_list]

    #load semantic role labeler
    predictor = Predictor.from_path("https://storage.googleapis.com/allennlp-public-models/structured-prediction-srl-bert.2020.12.15.tar.gz")

    #empty list for story semantic subjects
    storyss = []

    for n in range(len(sent_list)):
    # for n in range(1):
        # print(n)

        labels = predictor.predict(
            sentence = sent_list[n]
        )
        
        # print(labels)

        for verb in labels['verbs']:
        # print("verb:",verb['verb'])
            ss = []
            tags = verb['tags']
            for index, e in enumerate(tags):
                if e=='B-ARG0' or e=='I-ARG0': #or 'ARG0' in e:
                        word = (labels['words'][index])
                        ss.append(word)

                # print(ss)
                
                if len(ss)>1:
                        semsubj = ' '.join(ss)
                elif ss:
                        semsubj = ss[0]
            if ss:
                storyss.append(semsubj.lower())
    
    return storyss

def ner_person(storypath):
    '''
    input: path to story text
    output: list of ner
    '''

    #load in raw texts
    f = open(storypath, 'r')
    doc = f.read()
    f.close()

    #remove newlines
    new = doc.replace('\n', ' ').replace("\'", '"')
    # print(new)

    #split doc into sentences using spacy 
    sentences = [i for i in nlp(new).sents]

    #convert back to text to feed into predictor
    sent_list = [i.text for i in sentences]
    sent_list = [i.strip() for i in sent_list]
    sent_list = [i for i in sent_list if i]

    #load ner predictor
    predictor = Predictor.from_path("https://storage.googleapis.com/allennlp-public-models/ner-elmo.2021-02-12.tar.gz")

    #empty list for ner
    storyner = []

    for n in range(len(sent_list)):
    # for n in range(1):

        labels = predictor.predict(
            sentence = sent_list[n]
        )

        ner = []

        tags = labels['tags']

        for index,e in enumerate(tags):

            if e=='B-PER' or e=='I-PER':
                word = (labels['words'][index])
                ner.append(word)
            if len(ner)>1:
                person = ' '.join(ner)
                storyner.append(person)
            if e=='U-PER':
                word = (labels['words'][index])
                storyner.append(word)
    
    return storyner

def dep_link(storypath):
    '''
    input: path to story text
    output: list of ner
    '''

    #load in raw texts
    f = open(storypath, 'r')
    doc = f.read()
    f.close()

    #remove newlines
    new = doc.replace('\n', ' ').replace("\'", '"')
    # print(new)

    #split doc into sentences using spacy 
    sentences = [i for i in nlp(new).sents]

    #convert back to text to feed into predictor
    sent_list = [i.text for i in sentences]
    sent_list = [i.strip() for i in sent_list]
    sent_list = [i for i in sent_list if i]

    #load ner predictor
    predictor = Predictor.from_path("https://storage.googleapis.com/allennlp-public-models/biaffine-dependency-parser-ptb-2020.04.06.tar.gz")

    #empty list for ner
    storydp = []

    for n in range(len(sent_list)):
    # for n in range(1):

        labels = predictor.predict(
            sentence = sent_list[n]
        )

        dp = []

        tags = labels['predicted_dependencies']

        for index,e in enumerate(tags):

            if e=='nsubj':
                word = (labels['words'][index])
                dp.append(word)
            if dp:
                storydp.append(word)
    
    return storydp

def triple(storypath):
    '''
    input: path to story text
    output: list of semantic subject for story
    '''

    #load in raw texts
    f = open(storypath, 'r')
    doc = f.read()
    f.close()

    #remove newlines
    new = doc.replace('\n', ' ').replace("\'", '"')
    # print(new)

    #split doc into sentences using spacy 
    sentences = [i for i in nlp(new).sents]

    #convert back to text to feed into predictor
    sent_list = [i.text for i in sentences]
    sent_list = [i.strip() for i in sent_list]
    sent_list = [i for i in sent_list if i]

    #load semantic role labeler
    predictor = Predictor.from_path("https://storage.googleapis.com/allennlp-public-models/openie-model.2020.03.26.tar.gz")

    #empty list for story semantic subjects
    storytp = []

    for n in range(len(sent_list)):
    # for n in range(1):
        # print(n)

        labels = predictor.predict(
            sentence = sent_list[n]
        )
        
        # print(labels)

        for verb in labels['verbs']:
        # print("verb:",verb['verb'])
            tp = []
            tags = verb['tags']
            for index, e in enumerate(tags):
                if e=='B-ARG0' or e=='I-ARG0': #or 'ARG0' in e:
                        word = (labels['words'][index])
                        tp.append(word)

                # print(ss)
                
                if len(tp)>1:
                        subj = ' '.join(tp)
                elif tp:
                        subj = tp[0]
            if tp:
                storytp.append(subj)
    
    return storytp
