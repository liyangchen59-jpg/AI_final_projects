import torch
from datasets import load_dataset
from transformers import BertTokenizerFast, BertForSequenceClassification


dataset = load_dataset("glue", "sst2")
validation_dataset = dataset["validation"]
tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
model = BertForSequenceClassification.from_pretrained("./bert-for-sst2", return_dict=True)
model.eval()

correct = []
wrong = []
all_results = []

for i, example in enumerate(validation_dataset, start=1):
    inputs = tokenizer(
        example["sentence"],
        truncation=True,
        padding="max_length",
        max_length=64,
        return_tensors="pt"
    )

    with torch.no_grad():
        logits = model(**inputs).logits
        pred = logits.argmax(dim=-1).item()

    item = {
        "index": i,
        "sentence": example["sentence"],
        "label": example["label"],
        "prediction": pred
    }
    all_results.append(item)

    if pred == example["label"]:
        correct.append(item)
    else:
        wrong.append(item)

with open("sst2_eval_results.txt", "w", encoding="utf-8") as f:
    f.write("SST-2 validation results\n")
    f.write(f"Total validation samples: {len(validation_dataset)}\n")
    f.write(f"Correct examples found: {len(correct)}\n")
    f.write(f"Wrong examples found: {len(wrong)}\n\n")

    for item in all_results:
        f.write(f"Example {item['index']}\n")
        f.write(item["sentence"] + "\n")
        f.write(f"Label: {item['label']}, Prediction: {item['prediction']}\n")
        f.write(f"Result: {'correct' if item['label'] == item['prediction'] else 'wrong'}\n\n")

print("SST-2 validation results")
print(f"Total validation samples: {len(validation_dataset)}")
print(f"Correct examples found: {len(correct)}")
print(f"Wrong examples found: {len(wrong)}")
print("Saved all results to sst2_eval_results.txt")

print("\nCorrect examples:\n")
for i, item in enumerate(correct[:10], start=1):
    print(f"Example {i} (dataset index {item['index']})")
    print(item["sentence"])
    print(f"Label: {item['label']}, Prediction: {item['prediction']}")
    print()

print("Wrong examples:\n")
for i, item in enumerate(wrong[:10], start=1):
    print(f"Example {i} (dataset index {item['index']})")
    print(item["sentence"])
    print(f"Label: {item['label']}, Prediction: {item['prediction']}")
    print()