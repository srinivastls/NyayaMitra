from transformers import pipeline
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
import pandas as pd

model_name = "Srinivastl/NyayaMitra"
model = pipeline("text2text-generation", model=model_name)



nltk.download("punkt")

def compute_bleu(reference, prediction):
    reference_tokens = [nltk.word_tokenize(reference.lower())]  # Tokenize reference
    prediction_tokens = nltk.word_tokenize(prediction.lower())  # Tokenize prediction
    smoothie = SmoothingFunction().method4  # Smoothing for better BLEU scores
    return sentence_bleu(reference_tokens, prediction_tokens, smoothing_function=smoothie)

def compute_rouge(reference, prediction):
    scorer = rouge_scorer.RougeScorer(["rouge-1", "rouge-2", "rouge-l"], use_stemmer=True)
    scores = scorer.score(reference, prediction)
    return {
        "ROUGE-1": scores["rouge-1"].fmeasure,
        "ROUGE-2": scores["rouge-2"].fmeasure,
        "ROUGE-L": scores["rouge-l"].fmeasure,
    }

from datasets import load_dataset

# Load a dataset (Example: legal dataset from Hugging Face)
dataset = load_dataset("lex_glue", "eurlex")  # Change to your dataset

# Select a subset for evaluation
samples = dataset["test"].select(range(10))  # Change as needed

results = []
for sample in samples:
    input_text = sample["text"]  # Modify based on dataset structure
    reference_text = sample["summary"]  # Modify based on dataset structure

    model_output = model(input_text, max_length=100, truncation=True)[0]["generated_text"]
    bleu_score = compute_bleu(reference_text, model_output)
    rouge_scores = compute_rouge(reference_text, model_output)

    results.append({
        "Input": input_text,
        "Reference": reference_text,
        "Generated": model_output,
        "BLEU": bleu_score,
        "ROUGE-1": rouge_scores["ROUGE-1"],
        "ROUGE-2": rouge_scores["ROUGE-2"],
        "ROUGE-L": rouge_scores["ROUGE-L"],
    })

df = pd.DataFrame(results)
