import os
import json
from datetime import datetime
from tqdm.auto import tqdm
import torch
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
)
from datasets import load_from_disk
from peft import PeftModel
from sacrebleu.metrics import BLEU, TER
from nltk.translate.meteor_score import meteor_score

def load_model(model_name):
    """Load model"""
    try:
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name,
            device_map="auto",
        )
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        raise

def load_peft_model(model_name, adapter_path):
    """Load a PEFT model with proper configuration."""
    try:
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name,
            device_map="auto",
            low_cpu_mem_usage=True
        )
        peft_model = PeftModel.from_pretrained(model, adapter_path)
        peft_model.merge_and_unload()
        return peft_model
    except Exception as e:
        print(f"Error loading model: {e}")
        raise

def evaluate_model(model, tokenizer, device, test_dataset, stage_name):
    try:
        results_dir = "./evaluation_results"
        os.makedirs(results_dir, exist_ok=True)

        model.eval()

        # Initialize BLEU and TER metrics
        bleu = BLEU()
        ter = TER()

        # Prepare data
        decoded_preds = []
        decoded_labels = []

        data_collator = DataCollatorForSeq2Seq(tokenizer, model, pad_to_multiple_of=8, return_tensors="pt")
        dataloader = DataLoader(
            test_dataset, 
            collate_fn=data_collator, 
            batch_size=10, # Change batch size based on GPU memory
            shuffle=True
        )

        print("Starting evaluation on the test split...")
        with tqdm(total=len(dataloader), desc="Evaluating") as pbar:
            with torch.no_grad():
                for batch in dataloader:
                    inputs = {k: v.to(device) for k, v in batch.items() if k != "labels"}
                    labels = batch["labels"].to(device)

                    outputs = model.generate(
                        **inputs,
                        decoder_start_token_id=tokenizer.lang_code_to_id["vi_VN"],
                        num_return_sequences=1,
                        num_beams=5,
                        early_stopping=True
                    )
                    pbar.update(1)

                preds = tokenizer.batch_decode(outputs, skip_special_tokens=True)
                labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        decoded_preds.extend(preds)
        decoded_labels.extend(labels)

        # Calculate metrics
        bleu_score = bleu.corpus_score(decoded_preds, [decoded_labels]).score
        ter_score = ter.corpus_score(decoded_preds, [decoded_labels]).score
        meteor_scores = [
            meteor_score([ref.split()], pred.split())
            for ref, pred in zip(decoded_labels, decoded_preds)
        ]
        avg_meteor_score = sum(meteor_scores) / len(meteor_scores)

        # Prepare results dictionary
        evaluation_results = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "stage": stage_name,
            "metrics": {
                "BLEU": bleu_score,
                "TER": ter_score,
                "METEOR": avg_meteor_score
            },
            "sample_predictions": [
                {
                    "source": label,
                    "prediction": pred
                } for label, pred in zip(decoded_labels[:5], decoded_preds[:5])
            ]
        }

        # Generate filename with timestamp
        filename = f"{stage_name}_evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        filepath = os.path.join(results_dir, filename)

        # Save results to JSON file
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(evaluation_results, f, ensure_ascii=False, indent=4)

        print(f"\nEvaluation Metrics:")
        print(f"BLEU score: {bleu_score:.2f}")
        print(f"TER score: {ter_score:.2f}")
        print(f"METEOR score: {avg_meteor_score:.4f}")

    except Exception as e:
        print(f"Error during evaluation: {e}")
        raise

if __name__ == "__main__":
    # Path to the tokenized dataset, result of MedEV
    medev_tokenized_path = "./tokenized_dataset/MedEV"
    medev_result_path = "./finetuned_model/MedEV"
    medev_datasets = load_from_disk(medev_tokenized_path)

    # Load tokenizer
    model_name = "vinai/vinai-translate-en2vi-v2"
    tokenizer = AutoTokenizer.from_pretrained(model_name, src_lang="en_XX", tgt_lang="vi_VN")

    # Load model
    model = load_model(model_name)
    # model = load_peft_model(model_name, medev_result_path)

    device = torch.device('cuda')

    # Evaluate on MedEV test set
    print("\nEvaluating on MedEV test set:")
    evaluate_model(model, tokenizer, device, medev_datasets["test"], "MedEV")