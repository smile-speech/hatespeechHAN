import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from nltk.tokenize import sent_tokenize


from keras.layers import GRU


class preprocessing():
    def __init__(self, MAX_SENTENCES, MAX_SENTENCE_LENGTH):
        self.MAX_SENTENCES = MAX_SENTENCES
        self.MAX_SENTENCE_LENGTH = MAX_SENTENCE_LENGTH
  
  
    def data_read(self):
        df = pd.read_csv('/root/corona/hatespeech/Data/data_waseem_3.csv',encoding = "ISO-8859-1")
        df = df[['hate','comment']]
        df = df.sample(frac=1).reset_index(drop=True)

        nohate_count = len(df[df['hate'] == 0])
        sexism_count = len(df[df['hate'] == 1])
        racism_count = len(df[df['hate'] == 2])

        self.nohate = df[df.hate == 0].index
        self.sexism = df[df.hate == 1].index
        self.racism = df[df.hate == 2].index

        self.nohate = np.random.choice(self.nohate,racism_count, replace=False)
        self.sexism = np.random.choice(self.sexism,racism_count, replace=False)
        self.racism = np.random.choice(self.racism,racism_count, replace=False)

        self.df = df


    def data_split(self):

        nohate_train = self.nohate[:1741]
        nohate_test = self.nohate[1741:]
        sexism_train = self.sexism[:1741]
        sexism_test = self.sexism[1741:]
        racism_train = self.racism[:1741]
        racism_test = self.racism[1741:]

        train = np.concatenate((nohate_train,sexism_train,racism_train))
        test = np.concatenate((nohate_test,sexism_test,racism_test))

        train_df = self.df.loc[train]
        test_df = self.df.loc[test]

        train_df = train_df.sample(frac=1).reset_index(drop=True)
        test_df = test_df.sample(frac=1).reset_index(drop=True)

        self.train_x_data =[]
        self.train_y_data =[]
        self.test_x_data =[]
        self.test_y_data =[]

        length=len(train_df)
        for i in range(length):
            self.train_x_data.append(train_df.loc[i].comment)
            self.train_y_data.append(train_df.loc[i].hate)
            
        length=len(test_df)
        for i in range(length):
            self.test_x_data.append(test_df.loc[i].comment)
            self.test_y_data.append(test_df.loc[i].hate)


    def build_dataset(self, x_data, y_data):
        max_sentences= self.MAX_SENTENCES
        max_sentence_length=self.MAX_SENTENCE_LENGTH
        nb_instances = len(x_data)
        X_data = np.zeros((nb_instances, max_sentences, max_sentence_length), dtype='int32')
        
        
        for i, review in enumerate(x_data):
            tokenized_sentences = self.doc2hierarchical(review)
                
            X_data[i] = tokenized_sentences[None, ...]
            
        nb_classes = len(set(y_data))
        Y_data = to_categorical(y_data, nb_classes)
        
        return X_data, Y_data


    def _tokenizer(self):
        self.tokenizer = Tokenizer()
        self.tokenizer.fit_on_texts(self.train_x_data)
        self.tokenizer.fit_on_texts(self.test_x_data)

        self.max_nb_words = len(self.tokenizer.word_index) + 1

        train_X_data, train_Y_data = self.build_dataset(self.train_x_data, self.train_y_data)
        test_X_data, test_Y_data = self.build_dataset(self.test_x_data, self.test_y_data)
        train_X_data, val_X_data, train_Y_data, val_Y_data = train_test_split(train_X_data, train_Y_data, 
                                                                      test_size=0.1111, 
                                                                      random_state=42)

        return self.max_nb_words, self.tokenizer, train_X_data, val_X_data, train_Y_data, val_Y_data, self.test_x_data, self.test_y_data,test_X_data, test_Y_data


    def doc2hierarchical(self, text):
        sentences = sent_tokenize(text)
        max_sentences = self.MAX_SENTENCES
        max_sentence_length = self.MAX_SENTENCE_LENGTH
        tokenized_sentences = self.tokenizer.texts_to_sequences(sentences)
        tokenized_sentences = pad_sequences(tokenized_sentences, maxlen=max_sentence_length)

        pad_size = max_sentences - tokenized_sentences.shape[0]

        if pad_size <= 0:  # tokenized_sentences.shape[0] < max_sentences
            tokenized_sentences = tokenized_sentences[:max_sentences]
        else:
            tokenized_sentences = np.pad(
                tokenized_sentences, ((0, pad_size), (0, 0)),
                mode='constant', constant_values=0
            )
        
        return tokenized_sentences