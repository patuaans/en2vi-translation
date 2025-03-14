# Machine Translation Project: English to Vietnamese

This project focuses on training a sequence-to-sequence (seq2seq) model to translate English medical text to Vietnamese. It leverages state-of-the-art techniques, including the Transformer architecture, PEFT (Parameter-Efficient Fine-Tuning), and 4-bit quantization.

## Project Structure

The project consists of the following main files:

*   **`preprocess.py`:** Handles data preprocessing, including tokenization and dataset preparation.
*   **`train.py`:** Contains the code for training the translation model.
*   **`evaluate.py`:**  Evaluates the trained model using metrics like BLEU, TER, and METEOR.
*   **`translate.py`:** Performs translation of input text using the trained model.

## Running Environment

This project is designed and tested to run on **Ubuntu Linux**. The following specifics are important:

*   **Operating System:** Ubuntu Linux (tested on recent versions)
*   **Python 3.10:** Ensure you have Python 3.10 installed on your system.
*   **CUDA 12.2:** This project was developed and tested using CUDA 12.2. If you have a different CUDA version, you'll need to install the appropriate PyTorch version as outlined in the Installation section below.
*   **`venv`:**  It's strongly recommended to use a virtual environment (`venv`) to manage dependencies and avoid library conflicts.

## Installation

1. **Update System Packages and Install `venv`:**

    ```bash
    sudo apt-get update
    sudo apt install python3.10-venv -y
    ```

2. **Create and Activate Virtual Environment:**

    ```bash
    python3.10 -m venv venv  # Create a virtual environment named 'venv'
    source venv/bin/activate   # Activate the virtual environment
    ```

3. **Install Dependencies (CUDA 12.4):**

    If you have CUDA 12.4, install the dependencies using the following commands:

    ```bash
    pip install datasets transformers tiktoken protobuf blobfile sentencepiece peft bitsandbytes nltk evaluate wandb sacrebleu
    pip install torch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 --index-url https://download.pytorch.org/whl/cu124
    ```

4. **Install Dependencies (Other CUDA Versions):**

    If you have a different CUDA version, you need to install the correct PyTorch version compatible with your CUDA installation.
    *   Visit the PyTorch Previous Versions page: [https://pytorch.org/get-started/previous-versions](https://pytorch.org/get-started/previous-versions)
    *   Find the appropriate `pip install` command for your CUDA version and the desired PyTorch version (it's recommended to use a PyTorch version compatible with Python 3.10 if possible).
    *   **Replace** the `pip install torch torchvision torchaudio` command in step 3 with the command you found on the PyTorch website.
    *   **Example (for CUDA 11.7):**
        ```bash
        pip install datasets transformers tiktoken protobuf blobfile sentencepiece peft bitsandbytes sacrebleu nltk
        pip install torch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 --index-url https://download.pytorch.org/whl/cu117
        ```

5. **Download NLTK Wordnet:**

    ```bash
    python -c "import nltk; nltk.download('wordnet')"
    ```

## Usage

1. **Preprocessing:**
    Run the preprocessing script to prepare the datasets:

    ```bash
    python preprocess.py
    ```

    *   **Note:** This script will create tokenized datasets in the `./tokenized_dataset/` directory. You might need to adjust the paths to your datasets within the script.

2. **Training:**
    Train the model using the `train.py` script:

    ```bash
    python train.py
    ```

    *   **Note:**  The training process will create checkpoints and save the fine-tuned model in the `./finetuned_model/` directory. You can modify training parameters (e.g., learning rate, number of epochs) within the `train.py` script.

3. **Evaluation:**
    The `train.py` script automatically evaluates the model on the MedEV test set during training. The evaluation results are saved in the `./evaluation_results/` directory. These results include:
    *   **BLEU, TER, and METEOR scores:** These metrics provide a quantitative assessment of the translation quality.

4. **Translation:**
    Run the `translate.py` script to translate text in CSV files:

    ```bash
    python translate.py
    ```

    * **Note:** The translation process will create translated files in `./output_translated_mimic_iii_MedEV` and `./output_translated_mimic_iii_mix`
## Expected Outputs

After running the training and translation scripts, you should expect the following outputs:

1. **`./evaluation_results/`:** This directory will contain JSON files with the evaluation results (BLEU, TER, METEOR) for the MedEV test set. Each file is named with a timestamp and the stage it represents (e.g., `MedEV_evaluation_20231027_143000.json`).

2. **`./output_translated_mimic_iii_MedEV/`:** This directory will hold the translated MIMIC-III dataset using the model fine-tuned only on the MedEV corpus.

3. **`./output_translated_mimic_iii_mix/`:** This directory will contain the translated MIMIC-III dataset using the model fine-tuned on both MedEV and MIMIC-III datasets.

## Important Notes

*   **Ubuntu Linux Environment:** This project is developed and tested on Ubuntu Linux. While it may work on other systems, compatibility is not guaranteed.
*   **CUDA Version:** Carefully follow the instructions for installing the correct PyTorch version based on your CUDA installation.
*   **Dataset Paths:** Make sure the paths to your datasets within the scripts are correct.
*   **Computational Resources:** Training large Transformer models can be computationally intensive. Ensure you have a machine with sufficient GPU memory and processing power.
*   **Virtual Environment:**  Always activate your virtual environment (`source venv/bin/activate`) before running any of the scripts.

## Troubleshooting

*   **`ModuleNotFoundError`:** If you encounter a `ModuleNotFoundError`, double-check that you have activated your virtual environment and installed all the required packages.
*   **CUDA Errors:** If you get CUDA-related errors, verify that your CUDA driver and PyTorch versions are compatible. Refer to the PyTorch documentation for troubleshooting CUDA issues.
*   **Out of Memory (OOM) Errors:** If you encounter OOM errors during training, try reducing the `per_device_train_batch_size` in the training arguments of `train.py`. You might also need to experiment with gradient accumulation steps.

Feel free to ask if you have any more questions. Good luck!