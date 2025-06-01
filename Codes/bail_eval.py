import json
import argparse
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

def preprocess(example, tokenizer, max_length=512):
    print(example)
    return tokenizer(
        example['text'],
        truncation=True,
        padding='max_length',
        max_length=max_length
    )

def predict(model, tokenizer, dataset, device, batch_size=16):
    dataloader = DataLoader(dataset, batch_size=batch_size)
    model.to(device)
    model.eval()

    predictions = {}

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Predicting"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            ids = batch['id']

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()

            for _id, pred in zip(ids, preds):
                predictions[_id] = int(pred)

    return predictions

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, required=True, help='Hugging Face model path or ID')
    parser.add_argument('--output_file', type=str, required=True, help='Where to save the submission JSON')
    parser.add_argument('--max_length', type=int, default=512)
    parser.add_argument('--batch_size', type=int, default=16)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("🔄 Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name)

    print("📥 Loading IL-TUR Bail test dataset...")
    raw_dataset = load_dataset("Exploration-Lab/IL-TUR", "cjpe",split="test", revision="script",trust_remote_code=True)
    print(raw_dataset.column_names)
    print("🧼 Tokenizing dataset...")
    tokenized_dataset = raw_dataset.map(
        lambda x: preprocess(x, tokenizer, args.max_length),
        batched=True
    )
    tokenized_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'id'])

    print("🚀 Running predictions...")
    predictions = predict(model, tokenizer, tokenized_dataset, device, args.batch_size)

    print(f"💾 Saving submission to {args.output_file}")
    submission = {"bail": predictions}
    with open(args.output_file, 'w', encoding='utf-8') as f:
        json.dump(submission, f, indent=2)

    print("✅ Done!")

if __name__ == "__main__":
    main()
