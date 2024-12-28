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
import gc

def load_model_and_tokenizer(model_name, is_fine_tuned, adapter_path=None):
    try:
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name, src_lang="en_XX", tgt_lang="vi_VN")
        
        # Load model
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name,
            device_map="auto",
        )
        
        if is_fine_tuned:
            if not adapter_path:
                raise ValueError(f"Adapter path must be provided for fine-tuned model '{model_name}'.")
            # Load PEFT model if fine-tuned
            model = PeftModel.from_pretrained(model, adapter_path)
            model.merge_and_unload()
        
        return model, tokenizer
    except Exception as e:
        model_type = "fine-tuned" if is_fine_tuned else "pre-trained"
        print(f"Error loading {model_type} model '{model_name}': {e}")
        raise

def evaluate_model(model, tokenizer, device, dataset, model_name, fine_tuned):
    try:
        model.eval()

        # Initialize BLEU and TER metrics
        bleu = BLEU()
        ter = TER()

        # Prepare data
        decoded_preds = []
        decoded_labels = []

        data_collator = DataCollatorForSeq2Seq(tokenizer, model, pad_to_multiple_of=8, return_tensors="pt")
        dataloader = DataLoader(
            dataset, 
            collate_fn=data_collator, 
            batch_size=12,  # Adjust based on GPU memory
            shuffle=False
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
                    labels_decoded = tokenizer.batch_decode(labels, skip_special_tokens=True)

                    decoded_preds.extend(preds)
                    decoded_labels.extend(labels_decoded)

                    # Cleanup
                    del batch, inputs, labels, outputs, preds, labels_decoded
                    torch.cuda.empty_cache()

        # Calculate metrics
        bleu_score = bleu.corpus_score(decoded_preds, [decoded_labels]).score
        ter_score = ter.corpus_score(decoded_preds, [decoded_labels]).score
        meteor_scores = [
            meteor_score([ref.split()], pred.split())
            for ref, pred in zip(decoded_labels, decoded_preds)
        ]
        avg_meteor_score = sum(meteor_scores) / len(meteor_scores) if meteor_scores else 0.0

        # Prepare results dictionary
        evaluation_results = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "model_name": model_name,
            "fine_tuned": fine_tuned,
            "dataset_length": len(dataset),            
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

        return evaluation_results

    except Exception as e:
        print(f"Error during evaluation of '{model_name}': {e}")
        raise

def save_results(evaluation_results, results_dir):
    try:
        model_name = evaluation_results["model_name"].replace("/", "_")  # Replace slashes to avoid filepath issues
        filename = f"{model_name}_evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        filepath = os.path.join(results_dir, filename)

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(evaluation_results, f, ensure_ascii=False, indent=4)

        print(f"\nEvaluation Metrics for '{evaluation_results['model_name']}':")
        print(f"BLEU score: {evaluation_results['metrics']['BLEU']:.2f}")
        print(f"TER score: {evaluation_results['metrics']['TER']:.2f}")
        print(f"METEOR score: {evaluation_results['metrics']['METEOR']:.4f}")

    except Exception as e:
        print(f"Error saving results for '{evaluation_results['model_name']}': {e}")
        raise

def run_evaluation():
    # Paths
    tokenized_path = "./tokenized_dataset/MedEV"
    results_dir = "./evaluation_results"

    # Load tokenized dataset
    try:
        tokenized = load_from_disk(tokenized_path)
    except Exception as e:
        print(f"Error loading dataset from '{tokenized_path}': {e}")
        return

    # Ensure results directory exists
    os.makedirs(results_dir, exist_ok=True)

    # Set device
    device = torch.device('cuda')

    # Lists of models
    pre_trained_model_names = [
        "vinai/vinai-translate-en2vi-v2"
    ]

    # Fine-tuned models with their associated adapter paths
    fine_tuned_models = [
        # ("vinai/vinai-translate-en2vi-v2", "./finetuned_model/vinai-translate-en2vi-v2"),
    ]

    # Combine models with their fine-tuned status and adapter paths
    models = [(model_name, False, None) for model_name in pre_trained_model_names] + \
             [(model_name, True, adapter_path) for model_name, adapter_path in fine_tuned_models]

    if not models:
        print("No models to evaluate. Please add model names to the pre_trained_model_names or fine_tuned_models lists.")
        return

    # Loop through models and evaluate
    for model_info in models:
        model_name, is_fine_tuned, adapter_path = model_info
        status = "Fine-tuned" if is_fine_tuned else "Pre-trained"
        print(f"\n{status} Model: {model_name}")

        try:
            # Check if model exists before loading
            if not model_name:
                print(f"Model name is empty. Skipping...")
                continue

            # For fine-tuned models, ensure adapter_path is provided
            if is_fine_tuned and not adapter_path:
                print(f"Adapter path not provided for fine-tuned model '{model_name}'. Skipping...")
                continue

            # Load model and tokenizer
            model, tokenizer = load_model_and_tokenizer(model_name, is_fine_tuned, adapter_path)
            model.to(device)

            # Evaluate the model
            evaluation_results = evaluate_model(
                model=model,
                tokenizer=tokenizer,
                device=device,
                dataset=tokenized["test"],  # Evaluate on a subset of the test set
                model_name=model_name,
                fine_tuned=is_fine_tuned
            )

            # Save the evaluation results
            save_results(evaluation_results, results_dir)

        except Exception as e:
            print(f"Failed to evaluate '{model_name}': {e}")
        
        finally:
            # Free GPU memory
            del model, tokenizer
            gc.collect()
            torch.cuda.empty_cache()
            print(f"Freed GPU memory after evaluating '{model_name}'.")

if __name__ == "__main__":
    run_evaluation()