# hatespeechHAN

### Dependencies
- Python 3.6.9
- tensorflow 2.2.0rc3
- Keras 2.4.3
- Ubuntu 18.04
### Install
```bash
$ git clone https://github.com/smile-speech/hatespeechHAN.git
$ cd hatespeechHAN
$ pip install -r requirements.txt
```
### Start
```bash
$ python main.py
```

### argparse Info
```bash
parser.add_argument('--ms', type=int, help='MAX_SENTENCES',default=8,required=False)
parser.add_argument('--msl', type=int, help='MAX_SENTENCE_LENGTH',default=20, required=False)
parser.add_argument('--em', type=int, help='embedding_dim',default=300, required=False)
parser.add_argument('--epochs', type=int, help='epochs',default=100, required=False)
parser.add_argument('--batch_size', type=int, help='batch-size',default=64, required=False)
parser.add_argument('--d',type=str, help="dataset folder name",default="waseem", required=False)
parser.add_argument('--cl',type=int, help="number of class",default=3, required=False)
parser.add_argument('--lr',type=int, help="learning rate",default=0.01, required=False)
parser.add_argument('--op',type=int, help="optimizer",default=0.01, required=False)
```