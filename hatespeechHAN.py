import pandas as pd
import numpy as np
import random
import argparse
import urllib
import csv
import os
import keras

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras.preprocessing import text
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import text_to_word_sequence
from keras.utils import to_categorical
from nltk.tokenize import sent_tokenize

from tensorflow.compat.v1.keras.layers import CuDNNGRU
#from tensorflow.keras import backend as K
from tensorflow.compat.v1.keras import backend as K
from keras.engine.topology import Layer
#from tensorflow.compat.v1.keras.layers import CuDNNGRU
from keras.layers import Input, Embedding, Dense
from keras.layers import Lambda, Permute, RepeatVector, Multiply
from keras.layers import Bidirectional, TimeDistributed
from keras.layers import GRU
from keras.layers import BatchNormalization, Dropout
from keras.models import Model, Sequential
from keras.callbacks import ModelCheckpoint

import matplotlib.pyplot as plt
from numpy import argmax
from sklearn.metrics import classification_report

from sklearn.metrics import confusion_matrix, classification_report

import seaborn as sns
tf.compat.v1.disable_eager_execution()
experimental_run_tf_function=False


image_path = '/root/corona/hatespeechHAN/images/'

parser = argparse.ArgumentParser()
parser.add_argument('--max_sen', type=int, default=8, help='MAX_SENTENCES')
parser.add_argument('--max_sen_len', type=int, default=20, help='MAX_SENTENCE_LENGTH')
args = parser.parse_args()

# print( args.max_sen)
MAX_SENTENCES = args.max_sen
MAX_SENTENCE_LENGTH = args.max_sen_len


class AttentionLayer(Layer):
    def __init__(self, attention_dim, **kwargs):
        self.attention_dim = attention_dim
        super(AttentionLayer, self).__init__(**kwargs)
    
    def build(self, input_shape):
        self.W = self.add_weight(name='Attention_Weight',
                                 shape=(input_shape[-1], self.attention_dim),
                                 initializer='random_normal',
                                 trainable=True)
        self.b = self.add_weight(name='Attention_Bias',
                                 shape=(self.attention_dim, ),
                                 initializer='random_normal',
                                 trainable=True)
        self.u = self.add_weight(name='Attention_Context_Vector',
                                 shape=(self.attention_dim, 1),
                                 initializer='random_normal',
                                 trainable=True)
        super(AttentionLayer, self).build(input_shape)
        
    def call(self, x):
        # refer to the original paper
        # link: https://www.cs.cmu.edu/~hovy/papers/16HLT-hierarchical-attention-networks.pdf
        u_it = K.tanh(K.dot(x, self.W) + self.b)
        a_it = K.dot(u_it, self.u)
        a_it = K.squeeze(a_it, -1)
        a_it = K.softmax(a_it)
        
        return a_it
        
    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1])


class Hierarchical_attention_networks():
    

    def __init__(self, tokenizer, embedding_dim, max_nb_words,train_X_data, val_X_data, train_Y_data, val_Y_data):
        self.tokenizer = tokenizer
        self.embedding_dim = embedding_dim
        self.max_nb_words = max_nb_words
        self.train_X_data = train_X_data
        self.train_Y_data = train_Y_data
        self.val_X_data = val_X_data
        self.val_Y_data = val_Y_data


    def load_word2vec(self):

        embedding_dir = '/root/corona/hatespeech/embedding'
        from gensim.models import KeyedVectors
        embedding_path = os.path.join(embedding_dir, 'GoogleNews-vectors-negative300.bin')
        embeddings_index = KeyedVectors.load_word2vec_format(embedding_path, binary=True)
        
        return embeddings_index
        
    def load_embedding(self, embedding_type='word2vec'):
        
        if embedding_type == 'word2vec':
            embeddings_index = self.load_word2vec()
            
        self.embedding_matrix = np.random.normal(0, 1, (self.max_nb_words, self.embedding_dim))
        for word, i in self.tokenizer.word_index.items():
            try:
                embedding_vector = embeddings_index[word]
            except KeyError:
                embedding_vector = None
            if embedding_vector is not None:
                self.embedding_matrix[i] = embedding_vector
                
        return self.embedding_matrix


    # print("embedding_matrix.shape: {}".format(embedding_matrix.shape))


    def WeightedSum(self,attentions, representations):
        # from Shape(batch_size, len_units) to Shape(batch_size, rnn_dim * 2, len_units)
        repeated_attentions = RepeatVector(K.int_shape(representations)[-1])(attentions)
        # from Shape(batch_size, rnn_dim * 2, len_units) to Shape(batch_size, len_units, lstm_dim * 2)
        repeated_attentions = Permute([2, 1])(repeated_attentions)

        # compute representation as the weighted sum of representations
        aggregated_representation = Multiply()([representations, repeated_attentions])
        aggregated_representation = Lambda(lambda x: K.sum(x, axis=1))(aggregated_representation)

        return aggregated_representation
        

    def HAN_layer(self,
            nb_classes,
            embedding_dim=300,
            attention_dim=100,
            rnn_dim=50,
            include_dense_batch_normalization=False,
            include_dense_dropout=True,
            nb_dense=1,
            dense_dim=300,
            dense_dropout=0.2,
            optimizer = keras.optimizers.Adagrad(lr=0.01, epsilon=1e-6)):

        max_sentence_length = MAX_SENTENCE_LENGTH
        max_sentences = MAX_SENTENCES
        # embedding_matrix = (max_nb_words + 1, embedding_dim)
        #추가
        self.embedding_matrix = self.load_embedding('word2vec')
        max_nb_words = self.embedding_matrix.shape[0] - 1
        embedding_layer = Embedding(max_nb_words + 1, 
                                    embedding_dim,
                                    weights=[self.embedding_matrix],
                                    input_length=max_sentence_length,
                                    trainable=False)

        # first, build a sentence encoder
        sentence_input = Input(shape=(max_sentence_length, ), dtype='int32')
        embedded_sentence = embedding_layer(sentence_input)
        embedded_sentence = Dropout(dense_dropout)(embedded_sentence)
        contextualized_sentence = Bidirectional(GRU(rnn_dim, return_sequences=True))(embedded_sentence)
        
        # word attention computation
        word_attention = AttentionLayer(attention_dim)(contextualized_sentence)
        sentence_representation = self.WeightedSum(word_attention, contextualized_sentence)
        
        sentence_encoder = Model(inputs=[sentence_input], 
                                outputs=[sentence_representation])

        # then, build a document encoder
        document_input = Input(shape=(max_sentences, max_sentence_length), dtype='int32')
        embedded_document = TimeDistributed(sentence_encoder)(document_input)
        contextualized_document = Bidirectional(GRU(rnn_dim, return_sequences=True))(embedded_document)
        
        # sentence attention computation
        sentence_attention = AttentionLayer(attention_dim)(contextualized_document)
        document_representation = self.WeightedSum(sentence_attention, contextualized_document)
        
        # finally, add fc layers for classification
        fc_layers = Sequential()
        for _ in range(nb_dense):
            if include_dense_batch_normalization == True:
                fc_layers.add(BatchNormalization())
            fc_layers.add(Dense(dense_dim, activation='relu'))
            if include_dense_dropout == True:
                fc_layers.add(Dropout(dense_dropout))
        fc_layers.add(Dense(nb_classes, activation='softmax'))
        
        pred_sentiment = fc_layers(document_representation)

        model = Model(inputs=[document_input],
                    outputs=[pred_sentiment])
        
        ############### build attention extractor ###############
        word_attention_extractor = Model(inputs=[sentence_input],
                                        outputs=[word_attention])
        word_attentions = TimeDistributed(word_attention_extractor)(document_input)
        attention_extractor = Model(inputs=[document_input],
                                        outputs=[word_attentions, sentence_attention])
        
        model.compile(loss=['categorical_crossentropy'],
                optimizer=optimizer,
                metrics=['accuracy'])
        
        return model, attention_extractor


    def training(self):
        
        save_folder = os.path.join("/root/corona/hatespeechHAN/models")
        if not os.path.isdir(save_folder):
            os.mkdir(save_folder)
        model_path = os.path.join(save_folder, "modell.h5")

        checkpointer = ModelCheckpoint(filepath=model_path,
                                       monitor='val_acc',
                                       verbose=True,
                                       save_best_only=True,
                                       mode='max')

        # self.embedding_matrix = self.load_embedding('word2vec')#추가
        model, attention_extractor = self.HAN_layer(
                                            nb_classes=3,
                                            embedding_dim=300,
                                            attention_dim=100,
                                            rnn_dim=50,
                                            include_dense_batch_normalization=False,
                                            include_dense_dropout=True,
                                            nb_dense=1,
                                            dense_dim=300,
                                            dense_dropout=0.2,
                                            optimizer = keras.optimizers.Adagrad(lr=0.01, epsilon=1e-6))


        history = model.fit(x=[self.train_X_data],
                            y=[self.train_Y_data],
                            batch_size=64,
                            epochs=100,
                            verbose=True,
                            validation_data=(self.val_X_data, self.val_Y_data),
                            callbacks=[checkpointer])

        return history
        # print(history.history.keys())


# # # print(history.history.keys())



# # summarize history for accuracy
# plt.plot(history.history['accuracy'])
# plt.plot(history.history['val_accuracy'])
# plt.title('model accuracy')
# plt.ylabel('accuracy')
# plt.xlabel('epoch')
# plt.legend(['train', 'test'], loc='upper left')
# plt.show()


# # # summarize history for loss
# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
# plt.title('model loss')
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.legend(['train', 'test'], loc='upper left')
# plt.show()


# score = model.evaluate(test_X_data, test_Y_data, verbose=0, batch_size=64)
# # print("Test Accuracy of {}: {}".format(model_path, score[1]))



# length = len(test_x_data)
# y_true = test_y_data
# y_pred = []
# y_predict = model.predict(test_X_data)

# for i in range(length):
#     y_pred.append(argmax(y_predict[i]))

# target_names = ['0', '1','2']
# # print(classification_report(y_true, y_pred, target_names=target_names))



# comfmat = pd.DataFrame(confusion_matrix(y_true, y_pred), index=['nohate','sexism','racism'],columns=['nohate','sexism','racism'])



# word_rev_index={}
# for word, i in tokenizer.word_index.items():
#     word_rev_index[i] = word

# def sentiment_analysis(review):        
#     tokenized_sentences = doc2hierarchical(review)
    
#     # word attention만 가져오기
#     pred_attention = attention_extractor.predict(np.asarray([tokenized_sentences]))[0][0]
#     sent_attention = attention_extractor.predict(np.asarray([tokenized_sentences]))[1][0]
#     print(sent_attention)
#     sent_att_labels=[]
#     for sent_idx, sentence in enumerate(tokenized_sentences):
#         if sentence[-1] == 0:
#             continue
#         sent_len = sent_idx
#         sent_att_labels.append("Sentance "+str(sent_idx+1))
#     sent_att = sent_attention[0:sent_len+1]
#     sent_att = np.expand_dims(sent_att, axis=0)
#     sent_att_labels = np.expand_dims(sent_att_labels, axis=0) 

#     for sent_idx, sentence in enumerate(tokenized_sentences):
#         if sentence[-1] == 0:
#             continue
        
#         for word_idx in range(MAX_SENTENCE_LENGTH):
#             if sentence[word_idx] != 0:
#                 words = [word_rev_index[word_id] for word_id in sentence[word_idx:]]
#                 pred_att = pred_attention[sent_idx][-len(words):]
#                 pred_att = np.expand_dims(pred_att, axis=0)
#                 break

        
#         fig, ax = plt.subplots(figsize=(1,1))
#         plt.rc('xtick', labelsize=16)
#         #cmap="Blues",cmap='YlGnBu"
#         heatmap = sns.heatmap([[sent_att[0][sent_idx]]], xticklabels=False, yticklabels=False,cbar = False , annot=[[sent_att_labels[0][sent_idx]]],fmt ='', square=True, linewidths=0.1, cmap='coolwarm', center=0, vmin=0, vmax=1)
#         plt.xticks(rotation=45)
#         plt.show()

        
        
#         fig, ax = plt.subplots(figsize=(len(words), 2))
#         plt.rc('xtick', labelsize=16)
#         pred_att
#         word_list = np.expand_dims(words, axis=0)
#         heatmap = sns.heatmap(pred_att, xticklabels=False, yticklabels=False,cbar=False, square=True,annot=word_list ,fmt ='', annot_kws={"alpha":1,'rotation':15},cmap ="coolwarm_r", linewidths=0.2, center=0, vmin=0, vmax=1)
#         plt.xticks(rotation=45)
#         plt.show()
#         # plt.savefig(image_path+'/attention_img_sentence'+sent_idx+'_word'+word_idx+'.png')


# text =  "== Dear Yandman == Fuck you, do not censor me, cuntface. I think my point about French people being smelly frogs is very valid, it is not a matter of opinion. You go to hell you dirty bitch. Hugs and kisses Your secret admirer "
# sentiment_analysis(text)
