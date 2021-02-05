
import os
import torch
from models import InferSent

CWD = os.getcwd()
VERSION = 2
USE_CUDA = torch.cuda.is_available()

MODEL_PATH = f"{CWD}/encoder/infersent{VERSION}.pkl" 
VECTOR_GLOVE = f"{CWD}/vectors/glove.840B.300d.txt"
VECTOR_FASTTEXT = f"{CWD}/vectors/crawl-300d-2M.vec"
VECTOR_PATH = VECTOR_GLOVE if VERSION == 1 else VECTOR_FASTTEXT
VOCAB_SIZE= 100000

sentences = [
    'Yesterday, Muhyiddin Yassin announced that the Cabinet had agreed to appoint Khairy as the minister in charge of the Covid-19 immunisation programme.',
    'The DAP political education director said Khairy appears to be “much more” competent than Adham.'
]

def init_models(vocal_size: int=VOCAB_SIZE):
    model = InferSent({
        'bsize': 64, 
        'word_emb_dim': 300, 
        'enc_lstm_dim': 2048,
        'pool_type': 'max', 
        'dpout_model': 0.0, 
        'version': VERSION
    })
    model.load_state_dict(torch.load(MODEL_PATH))
    model = model.cuda() if USE_CUDA else model

    model.set_w2v_path(VECTOR_PATH)
    model.build_vocab_k_words(K=VOCAB_SIZE)
    return model

if __name__ == "__main__":
    model = init_models()

    for sentence in sentences:
        _, topWords = model.get_importance(sentence, tokenize=True)
        print(topWords)