import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from nltk.tokenize import sent_tokenize


from keras.layers import GRU


class preprocessing():
    def __init__(self, MAX_SENTENCES, MAX_SENTENCE_LENGTH,dataset,number_of_class):
        self.MAX_SENTENCES = MAX_SENTENCES
        self.MAX_SENTENCE_LENGTH = MAX_SENTENCE_LENGTH
        self.dataset = dataset
        self.number_of_class = number_of_class
  

            
    def data_ready(self):        
        # 3 class : waseem, hatebase
        # waseem: nohate(0), sexism(1), racism(2)
        # hatebase: hate(0). offensive(1), nohate(2)
        if self.number_of_class == 3:
            df = pd.read_csv('./data/'+self.dataset + '/data_'+ self.dataset+'_'+str(self.number_of_class)+'.csv',encoding = "ISO-8859-1")
            self.df = df[['hate','comment']].sample(frac=1).reset_index(drop=True)
            self.class1 = self.df[self.df.hate == 0].index
            self.class2 = self.df[self.df.hate == 1].index
            self.class3 = self.df[self.df.hate == 2].index

            #random sampling
            self.shortest = int(min(len(self.class1), len(self.class2),len(self.class3)))
            self.class1 = np.random.choice(self.class1,self.shortest, replace=False)
            self.class2 = np.random.choice(self.class2,self.shortest, replace=False)
            self.class3 = np.random.choice(self.class3,self.shortest, replace=False)

            self.train_size = int(len(self.class1)*0.9)
            
            #split into train,test each class
            class1_train = self.class1[:self.train_size]
            class1_test = self.class1[self.train_size:]
            class2_train = self.class2[:self.train_size]
            class2_test = self.class2[self.train_size:]
            class3_train = self.class3[:self.train_size]
            class3_test = self.class3[self.train_size:]
            #combine index
            train = np.concatenate((class1_train,class2_train,class3_train))
            test = np.concatenate((class1_test,class2_test,class3_test))
            
            #index to data & reset index
            train_df = self.df.loc[train].sample(frac=1).reset_index(drop=True)
            test_df = self.df.loc[test].sample(frac=1).reset_index(drop=True)

            #text & label split
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

        # 2 class : stormfront, wikipedia, kaggle
        # nohate(0), hate(1)
        if self.number_of_class == 2:
            df = pd.read_csv('./data/'+self.dataset + '/data_'+ self.dataset+'.csv',encoding = "ISO-8859-1")
            self.df = df[['hate','comment']].sample(frac=1).reset_index(drop=True)
            self.class1 = self.df[self.df.hate == 0].index
            self.class2 = self.df[self.df.hate == 1].index

            #random sampling
            self.shortest = int(min(len(self.class1), len(self.class2)))
            self.class1 = np.random.choice(self.class1,self.shortest, replace=False)
            self.class2 = np.random.choice(self.class2,self.shortest, replace=False)

            self.train_size = int(len(self.class1)*0.9)
            
            #split index into train,test
            class1_train = self.class1[:self.train_size]
            class1_test = self.class1[self.train_size:]
            class2_train = self.class2[:self.train_size]
            class2_test = self.class2[self.train_size:]

            #combine index
            train = np.concatenate((class1_train,class2_train))
            test = np.concatenate((class1_test,class2_test))
            
            #index to data & reset index
            train_df = self.df.loc[train].sample(frac=1).reset_index(drop=True)
            test_df = self.df.loc[test].sample(frac=1).reset_index(drop=True)

            #text & label split
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