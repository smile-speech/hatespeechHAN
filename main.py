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
    parser.add_argument('--d',type=str, help="dataset folder name",default="waseem", required=False)
    parser.add_argument('--cl',type=int, help="number of class",default=3, required=False)
    parser.add_argument('--lr',type=int, help="learning rate",default=0.01, required=False)
    parser.add_argument('--op',type=str, help="optimizer",default="Adagrad", required=False)
    
    args = parser.parse_args()

    MAX_SENTENCES = args.ms
    MAX_SENTENCE_LENGTH = args.msl
    embedding_dim = args.em
    epochs = args.epochs
    batch_size = args.batch_size
    lr = args.lr
    op = args.op

    #data name
    dataset = args.d
    number_of_class = args.cl

    pp = preprocessing(MAX_SENTENCES,MAX_SENTENCE_LENGTH,dataset,number_of_class)
    pp.data_ready()
    max_nb_words, tokenizer, train_X_data, val_X_data, train_Y_data, val_Y_data, test_x_data, test_y_data,test_X_data, test_Y_data= pp._tokenizer()
    han = Hierarchical_attention_networks(lr,op,epochs,batch_size,MAX_SENTENCES,MAX_SENTENCE_LENGTH,tokenizer, embedding_dim, max_nb_words,train_X_data, val_X_data, train_Y_data, val_Y_data,test_x_data,test_y_data,test_X_data, test_Y_data)
    han.training()
    han.evaluation()
    

    text = "== Dear Yandman == Fuck you, do not censor me, cuntface. I think my point about French people being smelly frogs is very valid, it is not a matter of opinion. You go to hell you dirty bitch. Hugs and kisses Your secret admirer "
    han.attention_visualization(text)
    


