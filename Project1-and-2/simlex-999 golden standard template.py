import os
import sys
import torch
from scipy import stats

os.environ['KMP_DUPLICATE_LIB_OK']='True'

'''
The parameters of functions and the implementation of each TODO part are not given and can be modified by yourself.
New functions can also be added if you want.
'''

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
EMBEDDING_DIR = os.path.join(BASE_DIR, "embedding_model")
RESULTS_DIR = os.path.join(BASE_DIR, "evaluation_results")
MODEL_NAME = "sgns" # cbow, skipgram, sgns
RESULT_NAME = "simlex.txt"
MODEL_PATH = os.path.join(EMBEDDING_DIR, f"{MODEL_NAME}.vec")
RESULTS_FILE = os.path.join(RESULTS_DIR, MODEL_NAME, RESULT_NAME)
if EMBEDDING_DIR not in sys.path:
    sys.path.append(EMBEDDING_DIR)

from utils import load_pretrained

os.makedirs(os.path.dirname(RESULTS_FILE), exist_ok=True)
with open(RESULTS_FILE, "w", encoding="utf-8"):
    pass


def log(*args):
    message = " ".join(str(arg) for arg in args)
    print(message)
    with open(RESULTS_FILE, "a", encoding="utf-8") as f:
        f.write(message + "\n")

# spearman correlation using function from scipy library. you can also implement by yourself.
def spearman(W, q):
    return torch.tensor(stats.spearmanr(q, W)[0])

def cos(v1, v2):
    return torch.dot(v1, v2) / (torch.norm(v1) * torch.norm(v2) + 1e-9)

vocab, embeds = load_pretrained(MODEL_PATH)
embed_vocabulary = set(vocab.token_to_idx)

# simlex-999 golden standard
with open(os.path.join(BASE_DIR, 'simlex-999.txt'),'r',encoding='utf-8') as f:
    lines = f.readlines()

standard = []
calculated = []
for line in lines:
    word1, word2, std = tuple(line.split())
    word1, word2, std = word1.lower(), word2.lower(), float(std)
    log("\nWord pair: {} and {}".format(word1, word2))

    if word1 in embed_vocabulary and word2 in embed_vocabulary:
        cal = (cos(embeds[vocab[word1]], embeds[vocab[word2]]).item() + 1) * 5
        log("The standard similarity is {} and the one calculated by embeddings is {}".format(std, cal))
        standard.append(std)
        calculated.append(cal)
    else:
        log("At least one of the words in this pair is not presented in the vocabulary.")

log("\nThe spearman correlation between the standard and calculated similarity is: {}".format(spearman(standard, calculated).item()))

