from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer

# Load tokenizer
model_name = "model_name"
tokenizer = AutoTokenizer.from_pretrained("vinai/vinai-translate-en2vi-v2", src_lang="en_XX", tgt_lang="vi_VN")

def load_and_prepare_dataset(data_path, is_csv=False):
    if is_csv:
        # Load MIMIC-III dataset from CSV
        dataset = load_dataset("csv", data_files=data_path)["train"]

        # Split the dataset
        full_split = dataset.train_test_split(test_size=0.1, seed=42)  # Separate 10% for test
        train_val_split = full_split["train"].train_test_split(test_size=0.2, seed=42)  # Split remaining 90% into 80% train, 20% validation

        # Combine splits into a DatasetDict
        dataset = DatasetDict({
            "train": train_val_split["train"],
            "validation": train_val_split["test"],
            "test": full_split["test"]
        })
    else:
        # Load MedEV dataset
        dataset = load_dataset(data_path)
        for split in dataset:
            dataset[split] = dataset[split].shuffle(seed=42)

    return dataset

# Preprocessing function
def preprocess_data(dataset, dataset_name):
    def tokenize_function(examples):
        # Handling different datasets (MedEV vs MIMIC-III)
        if dataset_name == "MedEV":
            inputs = [item["text"] for item in examples["en"]]
            targets = [item["text"] for item in examples["vi"]]
        elif dataset_name == "MIMIC-III Demo":
            inputs = examples["en"]
            targets = examples["vi"]
        else:
            raise ValueError(f"Unknown dataset name: {dataset_name}")

        # Tokenize inputs and targets
        model_inputs = tokenizer(
            inputs,
            targets,
            truncation=True,
        )
        return model_inputs
    
    # Apply tokenization to dataset
    tokenized_data = dataset.map(tokenize_function, batched=True)    
    tokenized_data = tokenized_data.remove_columns(["en", "vi"])
    tokenized_data = tokenized_data.with_format("torch")
    return tokenized_data

if __name__ == "__main__":
    # Define paths for MedEV and MIMIC-III datasets
    medev_path = "Angelectronic/MedEV"
    mimic_path = "./data/mimic-iii/MIMIC-III Demo.csv" 

    # Load the MedEV dataset from disk (no need for CSV loading here)
    medev_dataset = load_and_prepare_dataset(medev_path, is_csv=False)
    
    # Load the MIMIC-III dataset from CSV
    # mimic_dataset = load_and_prepare_dataset(mimic_path, is_csv=True)

    # Preprocess and tokenize each dataset
    tokenized_medev = preprocess_data(medev_dataset, dataset_name="MedEV")
    # tokenized_mimic = preprocess_data(mimic_dataset, dataset_name="MIMIC-III Demo")

    print("Tokenization completed for both datasets!")

    # Save tokenized datasets
    tokenized_medev.save_to_disk("./tokenized_dataset/MedEV")
    # tokenized_mimic.save_to_disk("./tokenized_dataset/MIMIC-III")
    print("Tokenized datasets saved.")
