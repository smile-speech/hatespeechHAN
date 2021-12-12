import pandas as pd
import urllib
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import numpy as np
import random
import argparse
from preprocessing import preprocessing
from hatespeechHAN import Hierarchical_attention_networks, AttentionLayer

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--ms', type=int, help='MAX_SENTENCES',default=8,required=False)
    parser.add_argument('--msl', type=int, help='MAX_SENTENCE_LENGTH',default=20, required=False)
    
    parser.add_argument('--em', type=int, help='embedding_dim',default=300, required=False)
    
    args = parser.parse_args()

    MAX_SENTENCES = args.ms
    MAX_SENTENCE_LENGTH = args.msl
    embedding_dim = args.em

    pp = preprocessing(MAX_SENTENCES,MAX_SENTENCE_LENGTH)
    pp.data_read()
    pp.data_split()
    max_nb_words, tokenizer, train_X_data, val_X_data, train_Y_data, val_Y_data = pp._tokenizer()
    # print(tokenizer)
    han = Hierarchical_attention_networks(tokenizer, embedding_dim, max_nb_words)
    embedding_matrix = han.load_embedding()
    print(embedding_matrix)


