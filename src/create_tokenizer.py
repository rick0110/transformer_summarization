import json
from datasets import load_dataset
from tqdm import tqdm
from tokenization import train_and_save_tokenizer


output_file = "./../data/HuggingFaceTB.json"

tokenizer_output_path = "./../data/HuggingFaceTB_tokenizer_bpe.json"
tokennn = train_and_save_tokenizer(
    dataset_path=output_file,
    text_columns=["prompt"],
    output_path=tokenizer_output_path,
    vocab_size=5000,
    model_type="bpe",
    max_examples=10000000
)



