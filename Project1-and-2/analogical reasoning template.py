import os
import random
import sys

import matplotlib.pyplot as plt
import torch

'''
The parameters of functions and the implementation of each TODO part are not given and can be modified by yourself.
New functions can also be added if you want.
'''

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
EMBEDDING_DIR = os.path.join(BASE_DIR, "embedding_model")
RESULTS_DIR = os.path.join(BASE_DIR, "evaluation_results")
MODEL_NAME = "skipgram" # cbow, skipgram, sgns
RESULT_NAME = "analogy.txt"
MODEL_PATH = os.path.join(EMBEDDING_DIR, f"{MODEL_NAME}.vec")
MODEL_RESULTS_DIR = os.path.join(RESULTS_DIR, MODEL_NAME)
PLOTS_DIR = os.path.join(MODEL_RESULTS_DIR, "analogy_plots")
RESULTS_FILE = os.path.join(MODEL_RESULTS_DIR, RESULT_NAME)
if EMBEDDING_DIR not in sys.path:
    sys.path.append(EMBEDDING_DIR)

from utils import load_pretrained

os.makedirs(os.path.dirname(RESULTS_FILE), exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)
with open(RESULTS_FILE, "w", encoding="utf-8"):
    pass


def log(*args):
    message = " ".join(str(arg) for arg in args)
    print(message)
    with open(RESULTS_FILE, "a", encoding="utf-8") as f:
        f.write(message + "\n")


def sanitize_filename(text):
    return "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in text)


def save_plot(example, plot_index):
    if example["predicted_word"] is None:
        return None

    predicted_vector = example["predicted_vector"].detach().cpu()
    result_vector = embeds[vocab[example["predicted_word"]]].detach().cpu()
    dimensions = range(1, predicted_vector.shape[0] + 1)
    filename = "{:02d}_{}_{}_{}_{}.png".format(
        plot_index,
        sanitize_filename(example["word_a"]),
        sanitize_filename(example["word_b"]),
        sanitize_filename(example["word_c"]),
        sanitize_filename(example["predicted_word"]),
    )
    plot_path = os.path.join(PLOTS_DIR, filename)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(dimensions, predicted_vector.tolist(), label="D_pred", linewidth=1.5)
    ax.plot(dimensions, result_vector.tolist(), label=example["predicted_word"], linewidth=1.5)
    ax.set_xlabel("Dimension")
    ax.set_ylabel("Value")
    ax.set_title(
        "{}:{}::{}:{}".format(
            example["word_a"],
            example["word_b"],
            example["word_c"],
            example["predicted_word"],
        )
    )
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(plot_path, dpi=200)
    plt.close(fig)

    return plot_path


# analogical reasoning task dataset
with open(os.path.join(BASE_DIR, 'analogical reasoning task.txt'),'r',encoding='utf-8') as f:
    lines = f.readlines()

vocab, embeds = load_pretrained(MODEL_PATH)
embed_vocabulary = set(vocab.token_to_idx)

def cos(W, x):
    return torch.matmul(W, x) / (torch.norm(W, dim=1) * torch.norm(x) + 1e-9)

def knn(W, x, k):
    similarities = cos(W, x)
    knn_result = similarities.topk(k=k)
    return knn_result.values.tolist(), knn_result.indices.tolist()

num_exist = 0
correct = []
evaluated_examples = []
random.seed(42)
for line in lines:
    line = line.strip()
    if not line or line.startswith(':'):
        continue

    words = line.split()
    if len(words) != 4:
        continue

    word_a, word_b, word_c, word_d = [word.lower() for word in words]
    if word_a not in embed_vocabulary or word_b not in embed_vocabulary or word_c not in embed_vocabulary or word_d not in embed_vocabulary:
        continue

    num_exist += 1
    vecs = embeds[vocab.convert_tokens_to_ids([word_a, word_b, word_c])]
    predicted = vecs[2] + vecs[1] - vecs[0]
    _, candidate_indices = knn(embeds, predicted, k=10)

    predicted_word = None
    filtered_candidates = []
    for index in candidate_indices:
        candidate = vocab.idx_to_token[index]
        if candidate in {word_a, word_b, word_c}:
            continue
        filtered_candidates.append(candidate)
        if predicted_word is None:
            predicted_word = candidate

    is_correct = predicted_word == word_d
    if is_correct:
        correct.append((word_a, word_b, word_c, word_d))

    evaluated_examples.append({
        "word_a": word_a,
        "word_b": word_b,
        "word_c": word_c,
        "word_d": word_d,
        "predicted_word": predicted_word,
        "predicted_vector": predicted.clone(),
        "is_correct": is_correct,
        "top_candidates": filtered_candidates[:5],
    })

sample_size = min(10, len(evaluated_examples))
plot_examples = []
if sample_size:
    sampled_examples = []

    if correct:
        correct_examples = [example for example in evaluated_examples if example["is_correct"]]
        sampled_examples.append(random.choice(correct_examples))
        remaining_size = sample_size - 1
        remaining_pool = [example for example in evaluated_examples if example is not sampled_examples[0]]
        if remaining_size > 0:
            sampled_examples.extend(random.sample(remaining_pool, remaining_size))
    else:
        sampled_examples = random.sample(evaluated_examples, sample_size)

    plot_examples = sampled_examples[:min(5, len(sampled_examples))]

    log("Random analogy examples ({})".format(sample_size))
    for idx, example in enumerate(sampled_examples, start=1):
        log("\nExample {}".format(idx))
        log("Analogy: {} : {} :: {} : ?".format(example["word_a"], example["word_b"], example["word_c"]))
        log("Expected:", example["word_d"])
        log("Predicted:", example["predicted_word"] if example["predicted_word"] is not None else "<none>")
        log("Result:", "Correct" if example["is_correct"] else "Incorrect")
        log(
            "Analysis: vec({}) - vec({}) + vec({}) is closest to \"{}\".".format(
                example["word_b"],
                example["word_a"],
                example["word_c"],
                example["predicted_word"] if example["predicted_word"] is not None else "<none>",
            )
        )
        log("Top candidates:", ", ".join(example["top_candidates"]) if example["top_candidates"] else "<none>")
else:
    log("Random analogy examples (0)")
    log("No valid evaluated examples were available.")

saved_plot_paths = []
for plot_index, example in enumerate(plot_examples, start=1):
    plot_path = save_plot(example, plot_index)
    if plot_path is not None:
        saved_plot_paths.append(plot_path)

if saved_plot_paths:
    log("\nSaved analogy plots:")
    for plot_path in saved_plot_paths:
        log(plot_path)

log("For {} analogy questions with all words presented in the vocabulary, {} of them are correctly answered."
    .format(num_exist, len(correct)))
