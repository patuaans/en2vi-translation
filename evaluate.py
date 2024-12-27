import os
import json
from datetime import datetime
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    BitsAndBytesConfig
)
from datasets import load_from_disk
from peft import PeftModel
from sacrebleu.metrics import BLEU, TER
from nltk.translate.meteor_score import meteor_score

def load_peft_model(model_name, adapter_path):
    """Load a PEFT model with proper configuration."""
    try:
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name,
            device_map="auto",
            low_cpu_mem_usage=True
        )



        # peft_model = PeftModel.from_pretrained(model, adapter_path)
        # peft_model.merge_and_unload()
        # return peft_model
        return model
    

    except Exception as e:
        print(f"Error loading model: {e}")
        raise

def evaluate_model(model_name, adapter_path, tokenizer, test_dataset, stage_name):
    try:
        results_dir = "./evaluation_results"
        os.makedirs(results_dir, exist_ok=True)

        # Load the fine-tuned model
        model = load_peft_model(model_name, adapter_path)
        model.eval()

        # Initialize BLEU and TER metrics
        bleu = BLEU()
        ter = TER()

        # Prepare data
        decoded_preds = []
        decoded_labels = []

        data_collator = DataCollatorForSeq2Seq(tokenizer, model, pad_to_multiple_of=8, return_tensors="pt")
        eval_dataloader = torch.utils.data.DataLoader(
            test_dataset, 
            collate_fn=data_collator, 
            batch_size=8,
            # shuffle=False
        )

        print("Starting evaluation on the test split...")
        for batch in eval_dataloader:
            inputs = {k: v.to("cuda") for k, v in batch.items() if k != "labels"}
            labels = batch["labels"].to("cuda")

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    decoder_start_token_id=tokenizer.lang_code_to_id["vi_VN"],
                    num_return_sequences=1,
                    num_beams=5,
                    early_stopping=True
                )

            preds = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

            decoded_preds.extend(preds)
            decoded_labels.extend(labels)

            if len(decoded_preds) <= 5:
                for i, (pred, label) in enumerate(zip(preds, labels)):
                    print(f"\nExample {i + 1}:")
                    print(f"Prediction: {pred}")
                    print(f"Reference: {label}")

            # Clean up
            del inputs, labels, outputs, preds, batch
            torch.cuda.empty_cache()

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

    # Load tokenizer
    model_name = "vinai/vinai-translate-en2vi-v2"
    tokenizer = AutoTokenizer.from_pretrained(model_name, src_lang="en_XX", tgt_lang="vi_VN")

    # Evaluate on MedEV test set
    print("\nEvaluating on MedEV test set:")
    medev_datasets = load_from_disk(medev_tokenized_path)
    evaluate_model(model_name, medev_result_path, tokenizer, medev_datasets["test"], "MedEV")