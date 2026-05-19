import torch
from datasets import load_dataset
from transformers import BertTokenizerFast, BertForSequenceClassification


dataset = load_dataset("glue", "mrpc")
validation_dataset = dataset["validation"]
tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
model = BertForSequenceClassification.from_pretrained("./bert-for-mrpc", return_dict=True)
model.eval()

correct = []
wrong = []
all_results = []

for i, example in enumerate(validation_dataset, start=1):
    inputs = tokenizer(
        example["sentence1"],
        example["sentence2"],
        truncation=True,
        padding="max_length",
        max_length=100,
        return_tensors="pt"
    )

    with torch.no_grad():
        logits = model(**inputs).logits
        pred = logits.argmax(dim=-1).item()

    item = {
        "index": i,
        "sentence1": example["sentence1"],
        "sentence2": example["sentence2"],
        "label": example["label"],
        "prediction": pred
    }
    all_results.append(item)

    if pred == example["label"]:
        correct.append(item)
    else:
        wrong.append(item)

with open("mrpc_eval_results.txt", "w", encoding="utf-8") as f:
    f.write("MRPC validation results\n")
    f.write(f"Total validation samples: {len(validation_dataset)}\n")
    f.write(f"Correct examples found: {len(correct)}\n")
    f.write(f"Wrong examples found: {len(wrong)}\n\n")

    for item in all_results:
        f.write(f"Example {item['index']}\n")
        f.write(item["sentence1"] + "\n")
        f.write(item["sentence2"] + "\n")
        f.write(f"Label: {item['label']}, Prediction: {item['prediction']}\n")
        f.write(f"Result: {'correct' if item['label'] == item['prediction'] else 'wrong'}\n\n")

print("MRPC validation results")
print(f"Total validation samples: {len(validation_dataset)}")
print(f"Correct examples found: {len(correct)}")
print(f"Wrong examples found: {len(wrong)}")
print("Saved all results to mrpc_eval_results.txt")

print("\nCorrect examples:\n")
for i, item in enumerate(correct[:10], start=1):
    print(f"Example {i} (dataset index {item['index']})")
    print(item["sentence1"])
    print(item["sentence2"])
    print(f"Label: {item['label']}, Prediction: {item['prediction']}")
    print()

print("Wrong examples:\n")
for i, item in enumerate(wrong[:10], start=1):
    print(f"Example {i} (dataset index {item['index']})")
    print(item["sentence1"])
    print(item["sentence2"])
    print(f"Label: {item['label']}, Prediction: {item['prediction']}")
    print()