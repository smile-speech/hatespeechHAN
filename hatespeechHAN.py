import pandas as pd
import numpy as np
from numpy import mean
from numpy import std
import os
import keras

import tensorflow as tf
from keras.preprocessing.sequence import pad_sequences
from nltk.tokenize import sent_tokenize

from tensorflow.compat.v1.keras import backend as K
from keras.engine.topology import Layer
from keras.layers import Input, Embedding, Dense
from keras.layers import Lambda, Permute, RepeatVector, Multiply
from keras.layers import Bidirectional, TimeDistributed
from keras.layers import GRU
from keras.layers import BatchNormalization, Dropout
from keras.models import Model, Sequential
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.utils import to_categorical

import matplotlib.pyplot as plt
from numpy import argmax
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.model_selection import train_test_split

import seaborn as sns
from pathlib import Path
tf.compat.v1.disable_eager_execution()
experimental_run_tf_function=False


class AttentionLayer(Layer):
    def __init__(self, attention_dim, **kwargs):
        self.attention_dim = attention_dim
        super(AttentionLayer, self).__init__(**kwargs)
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'attention_dim':self.attention_dim
        })
        return config

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
    
    def __init__(self,lr,op,MAX_SENTENCES,MAX_SENTENCE_LENGTH,embedding_dim, max_nb_words, tokenizer,number_of_class):
        self.MAX_SENTENCES = MAX_SENTENCES
        self.MAX_SENTENCE_LENGTH = MAX_SENTENCE_LENGTH
        self.tokenizer = tokenizer 
        self.embedding_dim = embedding_dim
        self.max_nb_words = max_nb_words
        self.embedding_matrix = self.load_embedding('word2vec')
        self.nb_classes = number_of_class
        self.lr = lr
        if op == 'Adagrad':
            self.optimizer = keras.optimizers.Adagrad(lr=self.lr, epsilon=1e-6)
        elif op =='Adadelta':
            self.optimizer = keras.optimizers.Adadelta(lr=self.lr, epsilon=1e-6)
        elif op == 'Adam':
            self.optimizer = keras.optimizers.Adam(lr=self.lr)
        elif op == 'RMSprop':
            self.optimizer = keras.optimizers.Adadelta(lr=self.lr,rho=0.9, epsilon=1e-6)


    def get_config(self):
        config = super().get_config()
        config.update({
            'MAX_SENTENCES':self.MAX_SENTENCES,
            'MAX_SENTENCE_LENGTH':self.MAX_SENTENCE_LENGTH,
            'embedding_dim':self.embedding_dim,
            'max_nb_words':self.max_nb_words,
            'tokenizer':self.tokenizer,
        })
        return config

    def load_word2vec(self):

        embedding_dir = './embedding'
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
            attention_dim=100,
            rnn_dim=50,
            include_dense_batch_normalization=False,
            include_dense_dropout=True,
            nb_dense=1,
            dense_dim=300,
            dense_dropout=0.2,
            ):
        
        # self.embedding_matrix = self.load_embedding('word2vec')
        self.max_nb_words = self.embedding_matrix.shape[0] - 1 ##수정
        embedding_layer = Embedding(self.max_nb_words + 1, 
                                    self.embedding_dim,
                                    weights=[self.embedding_matrix],
                                    input_length=self.MAX_SENTENCE_LENGTH,
                                    trainable=False)

        # first, build a sentence encoder
        sentence_input = Input(shape=(self.MAX_SENTENCE_LENGTH, ), dtype='int32')
        embedded_sentence = embedding_layer(sentence_input)
        embedded_sentence = Dropout(dense_dropout)(embedded_sentence)
        contextualized_sentence = Bidirectional(GRU(rnn_dim, return_sequences=True))(embedded_sentence)
        
        # word attention computation
        word_attention = AttentionLayer(attention_dim)(contextualized_sentence)
        sentence_representation = self.WeightedSum(word_attention, contextualized_sentence)
        
        sentence_encoder = Model(inputs=[sentence_input], 
                                outputs=[sentence_representation])

        # then, build a document encoder
        document_input = Input(shape=(self.MAX_SENTENCES, self.MAX_SENTENCE_LENGTH), dtype='int32')
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
        fc_layers.add(Dense(self.nb_classes, activation='softmax'))
        
        pred_sentiment = fc_layers(document_representation)

        self.model = Model(inputs=[document_input],
                    outputs=[pred_sentiment])
        
        ############### build attention extractor ###############
        word_attention_extractor = Model(inputs=[sentence_input],
                                        outputs=[word_attention])
        word_attentions = TimeDistributed(word_attention_extractor)(document_input)
        self.attention_extractor = Model(inputs=[document_input],
                                        outputs=[word_attentions, sentence_attention])
        
        self.model.compile(loss=['categorical_crossentropy'],
                optimizer = self.optimizer,
                metrics=['accuracy'])
        
        return self.model, self.attention_extractor

    def get_model_name(self,k):
        return '/model_'+str(k)+'.h5'

    def training(self, fold_var, dataset,epochs,batch_size,train_index,test_index, X_data, Y_data, x_data, y_data):     
        self.epochs = epochs
        self.batch_size = batch_size
        self.dataset = dataset
        
        self.X_data = X_data
        self.Y_data = Y_data
        self.x_data = x_data
        self.y_data = y_data

        ### Cross Validation (CV)
        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=100)
        self.fold_var = fold_var

        Path("./models/"+self.dataset+"_models(s="+str(self.MAX_SENTENCES)+"w="+str(self.MAX_SENTENCE_LENGTH)+")").mkdir(parents=True, exist_ok=True)
        save_folder = "./models/"+self.dataset+"_models(s="+str(self.MAX_SENTENCES)+"w="+str(self.MAX_SENTENCE_LENGTH)+")"
        
        if not os.path.isdir(save_folder):
            os.mkdir(save_folder)
        self.model_path = os.path.join(save_folder, "model.h5")

        train_val_X,test_X= self.X_data[train_index],self.X_data[test_index]
        train_val_Y,test_Y=self.y_data[train_index],self.y_data[test_index]
        
        self.nb_classes = len(set(train_val_Y))
        train_val_Y = to_categorical(train_val_Y, self.nb_classes)

        self.train_X, self.val_X, self.train_Y, self.val_Y= train_test_split(train_val_X, train_val_Y, 
                                                                    test_size=0.1111, 
                                                                    random_state=42)

        # self.embedding_matrix = self.load_embedding('word2vec')#추가
        self.model, self.attention_extractor = self.HAN_layer(
                                            attention_dim=100,
                                                rnn_dim=50,
                                                include_dense_batch_normalization=False,
                                                include_dense_dropout=True,
                                                nb_dense=1,
                                                dense_dim=300,
                                                dense_dropout=0.2)
        checkpointer = ModelCheckpoint(filepath=save_folder+self.get_model_name(self.fold_var),
                            monitor='val_loss',
                            verbose=True,
                            save_best_only=True,
                            mode='min')


        self.history = self.model.fit(x=[self.train_X],
                            y=[self.train_Y],
                            batch_size=self.batch_size,
                            epochs=self.epochs,
                            verbose=True,
                            validation_data=(self.val_X, self.val_Y),
                            callbacks=[es,checkpointer]
                            )

        self.model.load_weights(save_folder+"/model_"+str(self.fold_var)+".h5")
        length = len(test_Y)
        y_true = test_Y
        y_pred = []
        y_predict = self.model.predict(test_X)
        
        for i in range(length):
            y_pred.append(argmax(y_predict[i]))
        if self.nb_classes == 3:
            target_names = ['0','1','2']
        else:
            target_names = ['0','1']

        globals()['report{}'.format(self.fold_var)] =classification_report(y_true, y_pred, target_names=target_names)
        print(globals()['report{}'.format(self.fold_var)])
        globals()['report_dict{}'.format(self.fold_var)] = classification_report(y_true, y_pred, target_names=target_names, output_dict=True)
        print(globals()['report_dict{}'.format(self.fold_var)])

        tf.keras.backend.clear_session()

        print("-----Fold{}-----".format(self.fold_var))



    def evaluation(self):
        Test_accuracy =[]
        Test_precision =[]
        Test_recall =[]
        Test_f1 =[]
        for i in range(1,11):
            report_dict = globals()['report_dict{}'.format(i)]
            Test_accuracy.append(report_dict['accuracy'])
            Test_precision.append(report_dict['macro avg']['precision'])
            Test_recall.append(report_dict['macro avg']['recall'])
            Test_f1.append(report_dict['macro avg']['f1-score'])

        print('Test_Mean_Accuracy: %.4f(%.4f)' % (mean(Test_accuracy), std(Test_accuracy)))
        print('Test_Mean_Precision: %.4f(%.4f)' % (mean(Test_precision), std(Test_precision)))
        print('Test_Mean_Recall: %.4f(%.4f)' % (mean(Test_recall), std(Test_recall)))
        print('Test_Mean_F1: %.4f(%.4f)' % (mean(Test_f1), std(Test_f1)))
        
        # summarize history for accuracy
        plt.plot(self.history.history['accuracy'])
        plt.plot(self.history.history['val_accuracy'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')


        # # summarize history for loss
        plt.plot(self.history.history['loss'])
        plt.plot(self.history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')


    def doc2hierarchical(self, text):
        sentences = sent_tokenize(text)
        tokenized_sentences = self.tokenizer.texts_to_sequences(sentences)
        tokenized_sentences = pad_sequences(tokenized_sentences, maxlen=self.MAX_SENTENCE_LENGTH)

        pad_size = self.MAX_SENTENCES - tokenized_sentences.shape[0]

        if pad_size <= 0:  # tokenized_sentences.shape[0] < max_sentences
            tokenized_sentences = tokenized_sentences[:self.MAX_SENTENCES]
        else:
            tokenized_sentences = np.pad(
                tokenized_sentences, ((0, pad_size), (0, 0)),
                mode='constant', constant_values=0
            )
        
        return tokenized_sentences

    def attention_visualization(self,review):    
        word_rev_index={}
        for word, i in self.tokenizer.word_index.items():
            word_rev_index[i] = word    
        tokenized_sentences = self.doc2hierarchical(review)
        
        # word attention만 가져오기
        pred_attention = self.attention_extractor.predict(np.asarray([tokenized_sentences]))[0][0]
        sent_attention = self.attention_extractor.predict(np.asarray([tokenized_sentences]))[1][0]
        print(sent_attention)
        sent_att_labels=[]
        for sent_idx, sentence in enumerate(tokenized_sentences):
            if sentence[-1] == 0:
                continue
            sent_len = sent_idx
            sent_att_labels.append("Sentance "+str(sent_idx+1))
        sent_att = sent_attention[0:sent_len+1]
        sent_att = np.expand_dims(sent_att, axis=0)
        sent_att_labels = np.expand_dims(sent_att_labels, axis=0) 

        for sent_idx, sentence in enumerate(tokenized_sentences):
            if sentence[-1] == 0:
                continue
            
            for word_idx in range(self.MAX_SENTENCE_LENGTH):
                if sentence[word_idx] != 0:
                    words = [word_rev_index[word_id] for word_id in sentence[word_idx:]]
                    pred_att = pred_attention[sent_idx][-len(words):]
                    pred_att = np.expand_dims(pred_att, axis=0)
                    break

            
            fig, ax = plt.subplots(figsize=(1,1))
            plt.rc('xtick', labelsize=16)
            #cmap="Blues",cmap='YlGnBu"
            heatmap = sns.heatmap([[sent_att[0][sent_idx]]], xticklabels=False, yticklabels=False,cbar = False , annot=[[sent_att_labels[0][sent_idx]]],fmt ='', square=True, linewidths=0.1, cmap='coolwarm', center=0, vmin=0, vmax=1)
            plt.xticks(rotation=45)

            fig, ax = plt.subplots(figsize=(len(words), 2))
            plt.rc('xtick', labelsize=16)
            pred_att
            word_list = np.expand_dims(words, axis=0)
            heatmap = sns.heatmap(pred_att, xticklabels=False, yticklabels=False,cbar=False, square=True,annot=word_list ,fmt ='', annot_kws={"alpha":1,'rotation':15},cmap ="coolwarm_r", linewidths=0.2, center=0, vmin=0, vmax=1)
            plt.xticks(rotation=45)