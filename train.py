import torch
import math
import evaluate
import numpy as np
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    GenerationConfig,
    BitsAndBytesConfig,
    EarlyStoppingCallback,
    get_scheduler
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training
)
from torch.optim import AdamW
import wandb

# --- Configuration ---

# Model checkpoint
checkpoint = "VietAI/envit5-translation"
model_name = checkpoint.split('/')[-1]
result_path = "./finetuned_model/" + model_name

# Training parameters
num_epochs = 5
batch_size = 7
learning_rate = 5e-5
gradient_accumulation_steps = 4

# Initialize wandb
wandb.init(project="en2vn-translation", name=model_name)

# --- Data Loading and Preparation ---

def load_and_preprocess_data(checkpoint):
    """Loads the dataset, tokenizer, and preprocesses the data."""

    # Load dataset
    medev_dataset = load_dataset("Angelectronic/MedEV")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint, src_lang="en_XX", tgt_lang="vi_VN"
    )

    # Preprocessing function
    def preprocess_function(examples):
        inputs = [item["text"] for item in examples["en"]]
        targets = [item["text"] for item in examples["vi"]]
        model_inputs = tokenizer(
            inputs, text_target=targets, max_length=128, truncation=True
        )
        return model_inputs

    # Tokenize dataset
    tokenized_datasets = medev_dataset.map(preprocess_function, batched=True)
    tokenized_datasets = tokenized_datasets.remove_columns(["en", "vi"])
    return tokenized_datasets, tokenizer

# --- Evaluation Setup ---

# Load BLEU metric
metric = evaluate.load("sacrebleu")

def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]
    return preds, labels

def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

    result = metric.compute(predictions=decoded_preds, references=decoded_labels)
    result = {"bleu": result["score"]}

    prediction_lens = [
        np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds
    ]
    result["gen_len"] = np.mean(prediction_lens)
    result = {k: round(v, 4) for k, v in result.items()}
    return result

# --- Model Configuration and Initialization ---

def configure_and_initialize_model(checkpoint):
    """Configures quantization, LoRA, and loads the pre-trained model."""

    # BitsAndBytes configuration for 4-bit quantization
    nf4_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    # LoRA configuration
    peft_config = LoraConfig(
        task_type="SEQ_2_SEQ_LM",
        inference_mode=False,
        r=16,
        lora_alpha=32,
        lora_dropout=0.1,
        bias="none",
        target_modules="all-linear"
    )

    # Load pre-trained model with quantization
    model = AutoModelForSeq2SeqLM.from_pretrained(
        checkpoint,
        device_map="auto",
        quantization_config=nf4_config
    )

    device = torch.device("cuda")
    model = model.to(device)

    model.gradient_checkpointing_enable()
    # Prepare model for k-bit training (gradient checkpointing, etc.)
    model = prepare_model_for_kbit_training(model)

    # Apply PEFT with LoRA
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    return model

# --- Training Setup ---

def setup_training(model, tokenizer, tokenized_datasets):
    """Sets up the training arguments, data collator, optimizer, and trainer."""

    # Ensure tokenizer uses the EOS token as the padding token
    tokenizer.pad_token = tokenizer.eos_token

    # Data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer, 
        model, 
        pad_to_multiple_of=8, 
        return_tensors="pt"
    )
    
    # Update generation config
    # print(model.generation_config)
    # generation_config = GenerationConfig(
    #     max_length=1024,
    #     num_beams=5,
    #     bos_token_id=model.generation_config.bos_token_id,
    #     eos_token_id=model.generation_config.eos_token_id,
    #     pad_token_id=model.generation_config.pad_token_id,
    #     do_sample=True,
    # )

    # Calculate num_training_steps
    train_size = len(tokenized_datasets["train"].select(range(100)))
    batches_per_epoch = math.ceil(train_size / (batch_size * gradient_accumulation_steps))
    num_training_steps = batches_per_epoch * num_epochs

    # Set eval_steps to 20% of num_training_steps
    eval_steps = max(1, num_training_steps // 20)

    print(f"Training Size: {train_size}")
    print(f"Batches per Epoch: {batches_per_epoch}")
    print(f"Total Training Steps: {num_training_steps}")
    print(f"Evaluation Steps (10% of total): {eval_steps}")

    # Training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=result_path,
        report_to="wandb",  # Add this line to report to wandb
        eval_strategy="steps",
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,  
        gradient_checkpointing=True,
        learning_rate=learning_rate,
        eval_steps=eval_steps,
        save_steps=eval_steps,
        save_total_limit=3,
        num_train_epochs=num_epochs,
        warmup_steps=1250, 
        predict_with_generate=True,
        gradient_checkpointing_kwargs={'use_reentrant': False},
        fp16=True,
        load_best_model_at_end=True,
        metric_for_best_model="bleu",
        greater_is_better=True,
        logging_strategy="steps",
        logging_steps=eval_steps,
        push_to_hub=True,
        # logging_dir="./logs",
        # generation_config=generation_config,
    )

    # Optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=training_args.learning_rate,
        weight_decay=0.01  
    )

    # Scheduler
    lr_scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=int(num_training_steps * 0.1),
        num_training_steps=num_training_steps,
    )

    # Early stopping callback
    early_stopping = EarlyStoppingCallback(
        early_stopping_patience=10, 
        early_stopping_threshold=0.001
    )

    # Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=tokenized_datasets["train"].select(range(100)),
        eval_dataset=tokenized_datasets["validation"].select(range(10)),
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
        # callbacks=[early_stopping],
        optimizers=(optimizer, lr_scheduler), 
    )

    return trainer

# --- Main Execution ---

if __name__ == "__main__":
    tokenized_datasets, tokenizer = load_and_preprocess_data(checkpoint)
    model = configure_and_initialize_model(checkpoint)
    trainer = setup_training(model, tokenizer, tokenized_datasets)

    # Disable cache usage during training (to avoid warnings)
    model.config.use_cache = False

    # Train the model
    trainer.train(resume_from_checkpoint=False)

    # Re-enable cache usage after training
    model.config.use_cache = True

    # Save the PEFT adapter
    model.save_pretrained(result_path)

    model.push_to_hub(model_name)

    # Finish the wandb run
    wandb.finish()