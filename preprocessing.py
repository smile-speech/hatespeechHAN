import pandas as pd
import numpy as np

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.layers import GRU

from nltk.tokenize import sent_tokenize



class preprocessing():
    def __init__(self, MAX_SENTENCES, MAX_SENTENCE_LENGTH,dataset,number_of_class):
        self.MAX_SENTENCES = MAX_SENTENCES
        self.MAX_SENTENCE_LENGTH = MAX_SENTENCE_LENGTH
        self.dataset = dataset
        self.number_of_class = number_of_class
  

    def get_model_name(self):
        return '/model_'+str(self.k)+'.h5'     

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

            #######수정
            all_data = np.concatenate([self.class1,self.class2,self.class3])
            all_df = df.loc[all_data]
            all_df = all_df.sample(frac=1).reset_index(drop=True)

            self.x_data =[]
            self.y_data =[]

            length=len(all_df)
            for i in range(length):
                self.x_data.append(all_df.loc[i].comment)
                self.y_data.append(all_df.loc[i].hate)
                
            self.y_data = np.array(self.y_data)


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

            #######수정
            self.all_data = np.concatenate([self.class1,self.class2])
            self.all_df = df.loc[all_data]
            self.all_df = all_df.sample(frac=1).reset_index(drop=True)

            self.x_data =[]
            self.y_data =[]


            length=len(self.all_df)
            for i in range(length):
                self.x_data.append(self.all_df.loc[i].comment)
                self.y_data.append(self.all_df.loc[i].hate)
                
            self.y_data = np.array(self.y_data)



    def _tokenizer(self):

        ##추가
        self.tokenizer = Tokenizer()
        self.all_data = self.x_data
        self.tokenizer.fit_on_texts(self.all_data)

        count_thres = 0
        low_count_words = [w for w,c in self.tokenizer.word_counts.items() if c < count_thres]

        # print(len(self.tokenizer.word_index))
        for w in low_count_words:
            del self.tokenizer.word_index[w]
            del self.tokenizer.word_docs[w]
            del self.tokenizer.word_counts[w]

        self.max_nb_words = len(self.tokenizer.word_index) + 1

        self.X_data, self.Y_data = self.build_dataset(self.x_data, self.y_data)

        return self.max_nb_words, self.tokenizer, self.X_data, self.Y_data, self.x_data, self.y_data


    def doc2hierarchical(self, text):
        sentences = sent_tokenize(text)
        tokenized_sentences = self.tokenizer.texts_to_sequences(sentences)
        # tokenized_sentences = pad_sequences(tokenized_sentences, maxlen=self.MAX_SENTENCE_LENGTH)
        tokenized_sentences = pad_sequences(tokenized_sentences,padding='post', maxlen=self.MAX_SENTENCE_LENGTH,truncating="post")
             
        pad_size = self.MAX_SENTENCES - tokenized_sentences.shape[0]
        if pad_size <= 0:  # tokenized_sentences.shape[0] < max_sentences
            tokenized_sentences = tokenized_sentences[:self.MAX_SENTENCES]
        else:
            tokenized_sentences = np.pad(
                tokenized_sentences, ((0, pad_size), (0, 0)),
                mode='constant', constant_values=0
            )
        
        return tokenized_sentences


    def build_dataset(self, x_data, y_data):

        nb_instances = len(x_data)
        self.X_data = np.zeros((nb_instances, self.MAX_SENTENCES, self.MAX_SENTENCE_LENGTH), dtype='int32')
        
        
        for i, review in enumerate(x_data):
            tokenized_sentences = self.doc2hierarchical(review)
                
            self.X_data[i] = tokenized_sentences[None, ...]
            
        self.nb_classes = len(set(y_data))
        self.Y_data = to_categorical(y_data, self.nb_classes)
        
        return self.X_data, self.Y_data