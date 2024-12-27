import nltk
nltk.download('wordnet')
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
    BitsAndBytesConfig,
    EarlyStoppingCallback
)
from datasets import load_from_disk
from peft import (
    LoraConfig, 
    get_peft_model, 
)
from torch.optim import AdamW

def get_quantization_config():
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        # llm_int8_enable_fp32_cpu_offload=True,
    )

def train_model(model, tokenizer, tokenized_datasets, result_path):
    try:
        # PEFT configuration
        peft_config = LoraConfig(
            task_type= "SEQ_2_SEQ_LM",  
            inference_mode=False,               
            r=32,                               
            lora_alpha=64,                     
            lora_dropout=0.1,                  
            bias="none",                        
            target_modules="all-linear"
        )
        
        model = get_peft_model(model_nf4, peft_config)

        model.print_trainable_parameters()

        # Data collator for dynamic padding
        data_collator = DataCollatorForSeq2Seq(tokenizer, model, pad_to_multiple_of=8, return_tensors="pt")

        # Training arguments
        training_args = Seq2SeqTrainingArguments(
            output_dir=result_path,
            eval_strategy="steps",
            eval_steps=200,     
            save_steps=200,
            save_strategy="steps",
            learning_rate=5e-5, 
            save_total_limit=2,
            num_train_epochs=5,
            per_device_train_batch_size=20,  
            per_device_eval_batch_size=2,
            gradient_accumulation_steps=4,  
            warmup_steps=1250, 
            predict_with_generate=True,
            generation_max_length=256,
            fp16=True,
            # logging_steps=1,
	        # dataloader_num_workers=8,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            optim="paged_adamw_8bit"
        )

        # Define a custom optimizer
        optimizer = AdamW(
            model.parameters(), 
            lr=training_args.learning_rate, 
            weight_decay=0.01  
        )
        
        # Early stopping callback
        early_stopping = EarlyStoppingCallback(early_stopping_patience=3, early_stopping_threshold=0.01)

        # Trainer setup
        trainer = Seq2SeqTrainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_datasets["train"],
            eval_dataset=tokenized_datasets["validation"],
            processing_class=tokenizer,
            data_collator=data_collator,
            optimizers=(optimizer, None),
            callbacks=[early_stopping]
        )

        #Train model
        model.config.use_cache = False # silence the warnings
        trainer.train()
        
        # Renable warnings
        model.config.use_cache = True

        # Save the PEFT adapter
        model.save_pretrained(result_path)

    except Exception as e:
        print(f"Error during training: {e}")
        raise

if __name__ == "__main__":
    # Path to the tokenized dataset, result of MedEV
    medev_tokenized_path = "./tokenized_dataset/MedEV"
    medev_result_path = "./finetuned_model/MedEV"

    # Load tokenizer and model
    model_name = "vinai/vinai-translate-en2vi-v2"
    tokenizer = AutoTokenizer.from_pretrained(model_name, src_lang="en_XX", tgt_lang="vi_VN")

    nf4_config = get_quantization_config()
    
    model_nf4 = AutoModelForSeq2SeqLM.from_pretrained(
        model_name,
        quantization_config=nf4_config,
        device_map="auto",
        trust_remote_code=False,
        revision="main",
        low_cpu_mem_usage=True
    )

    # Stage 1: Train on MedEV dataset
    print("Stage 1: Training on MedEV dataset")
    medev_datasets = load_from_disk(medev_tokenized_path)
    train_model(model_nf4, tokenizer, medev_datasets, medev_result_path)