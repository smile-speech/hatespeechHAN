import argparse
from preprocessing import preprocessing
from hatespeechHAN import Hierarchical_attention_networks

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--ms', type=int, help='MAX_SENTENCES',default=8,required=False)
    parser.add_argument('--msl', type=int, help='MAX_SENTENCE_LENGTH',default=20, required=False)
    
    parser.add_argument('--em', type=int, help='embedding_dim',default=300, required=False)
    parser.add_argument('--epochs', type=int, help='epochs',default=100, required=False)
    parser.add_argument('--batch_size', type=int, help='batch-size',default=64, required=False)
    
    args = parser.parse_args()

    MAX_SENTENCES = args.ms
    MAX_SENTENCE_LENGTH = args.msl
    embedding_dim = args.em
    epochs = args.epochs
    batch_size = args.batch_size

    pp = preprocessing(MAX_SENTENCES,MAX_SENTENCE_LENGTH)
    pp.data_read()
    pp.data_split()
    max_nb_words, tokenizer, train_X_data, val_X_data, train_Y_data, val_Y_data, test_x_data, test_y_data,test_X_data, test_Y_data= pp._tokenizer()
    # print(tokenizer)
    han = Hierarchical_attention_networks(epochs,batch_size,MAX_SENTENCES,MAX_SENTENCE_LENGTH,tokenizer, embedding_dim, max_nb_words,train_X_data, val_X_data, train_Y_data, val_Y_data,test_x_data,test_y_data,test_X_data, test_Y_data)
    # embedding_matrix = han.load_embedding()
    # print("embedding_matrix.shape: {}".format(embedding_matrix.shape))
    han.training()

    han.evaluation()
    

    text =  "== Dear Yandman == Fuck you, do not censor me, cuntface. I think my point about French people being smelly frogs is very valid, it is not a matter of opinion. You go to hell you dirty bitch. Hugs and kisses Your secret admirer "
    han.attention_visualization(text)
    


