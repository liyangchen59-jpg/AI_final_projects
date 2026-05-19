import os
import sys
import torch

'''
The parameters of functions and the implementation of each TODO part are not given and can be modified by yourself.
New functions can also be added if you want.
'''

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
EMBEDDING_DIR = os.path.join(BASE_DIR, "embedding_model")
RESULTS_DIR = os.path.join(BASE_DIR, "evaluation_results")
MODEL_NAME = "skipgram" # cbow, skipgram, sgns
RESULT_NAME = "knn.txt"
MODEL_PATH = os.path.join(EMBEDDING_DIR, f"{MODEL_NAME}.vec")
RESULTS_FILE = os.path.join(RESULTS_DIR, MODEL_NAME, RESULT_NAME)
if EMBEDDING_DIR not in sys.path:
    sys.path.append(EMBEDDING_DIR)

from utils import load_pretrained

K = 10

os.makedirs(os.path.dirname(RESULTS_FILE), exist_ok=True)
with open(RESULTS_FILE, "w", encoding="utf-8"):
    pass


def log(*args):
    message = " ".join(str(arg) for arg in args)
    print(message)
    with open(RESULTS_FILE, "a", encoding="utf-8") as f:
        f.write(message + "\n")

# query word list
word_list = ["july", "reliable", "play", "willing", "good", "very", "patient", "concerned", "important", "powerful", "quickly", "generally", "gradually", "happy", "able", "close", "near",
             "saturday", "friend", "company", "road", "plane", "war", "politics", "building", "student", "university", "realm", "china", "experience", "police",
             "give", "create", "tell", "become", "lack", "win", "help", "gain", "get", "take", "use", "set", "find", "increase",
             "difficult", "go", "man", "ten", "year"]

vocab, embeds = load_pretrained(MODEL_PATH)
embed_vocabulary = set(vocab.token_to_idx)

# calculate cosine similarity
def cos(W, x):
    return torch.matmul(W, x) / (torch.norm(W, dim=1) * torch.norm(x) + 1e-9)

# find k-nearest neighbors
def knn(W, x, k):
    similarities = cos(W, x)
    knn_result = similarities.topk(k=k)
    return knn_result.values.tolist(), knn_result.indices.tolist()

query_average_similarities = []

# evaluation process for each word
def evaluate(word):
    knn_words = []
    knn_sims = []
    if word not in embed_vocabulary:
        log("\nQuery:", word)
        log("The query word is not presented in the vocabulary.")
        return

    knn_values, knn_indices = knn(embeds, embeds[vocab[word]], K + 1)
    for similarity, index in zip(knn_values, knn_indices):
        candidate = vocab.idx_to_token[index]
        if candidate == word:
            continue
        knn_words.append(candidate)
        knn_sims.append(similarity)
        if len(knn_words) == K:
            break

    log("\nQuery:", word)
    for i in range(len(knn_words)):
        log("The similarity of {} is {:.4f}.".format(knn_words[i], knn_sims[i]))

    average_similarity = sum(knn_sims) / len(knn_sims)
    query_average_similarities.append(average_similarity)
    log("Average similarity for {} is {:.4f}.".format(word, average_similarity))

for word in word_list:
    evaluate(word)

if query_average_similarities:
    overall_average_similarity = sum(query_average_similarities) / len(query_average_similarities)
    log("\nOverall average similarity across {} query words is {:.4f}.".format(len(query_average_similarities), overall_average_similarity))
