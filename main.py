import argparse
from preprocessing import preprocessing
from hatespeechHAN import Hierarchical_attention_networks
from sklearn.model_selection import KFold, StratifiedKFold

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--ms', type=int, help='MAX_SENTENCES',default=8,required=False)
    parser.add_argument('--msl', type=int, help='MAX_SENTENCE_LENGTH',default=20, required=False)
    parser.add_argument('--em', type=int, help='embedding_dim',default=300, required=False)
    parser.add_argument('--epochs', type=int, help='epochs',default=100, required=False)
    parser.add_argument('--batch_size', type=int, help='batch-size',default=64, required=False)
    parser.add_argument('--d',type=str, help="dataset folder name",default="waseem", required=False)
    parser.add_argument('--cl',type=int, help="number of class",default=3, required=False)
    parser.add_argument('--lr',type=float, help="learning rate",default=0.01, required=False)
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
    max_nb_words, tokenizer, X_data, Y_data, x_data, y_data= pp._tokenizer()
    print ("-----preprocessing completed-----")

    kf = KFold(n_splits=10)                        
    skf = StratifiedKFold(n_splits = 10, random_state = 7, shuffle = True)
    fold_var =1 
    ### Cross Validation (CV)
    for train_index, test_index in skf.split(X_data,y_data):
        han = Hierarchical_attention_networks(
            lr,
            op,
            MAX_SENTENCES,MAX_SENTENCE_LENGTH,
            embedding_dim, 
            max_nb_words, 
            tokenizer,
            number_of_class)
        han.training(fold_var,dataset,epochs,batch_size,train_index,test_index,X_data,Y_data,x_data,y_data)
        fold_var +=1
    
    han.evaluation()

    # text = "== Dear Yandman == Fuck you, do not censor me, cuntface. I think my point about French people being smelly frogs is very valid, it is not a matter of opinion. You go to hell you dirty bitch. Hugs and kisses Your secret admirer "
    # han.attention_visualization(text)
    


