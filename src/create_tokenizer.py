import json
from datasets import load_dataset
from tqdm import tqdm
from tokenization import train_and_save_tokenizer


output_file = "./../data/hh-rlhf.json"

tokenizer_output_path = "./../data/hh-rlhf_tokenizer_bpe.json"
tokennn = train_and_save_tokenizer(
    dataset_path=output_file,
    text_columns=["chosen"],
    output_path=tokenizer_output_path,
    vocab_size=2000,
    model_type="bpe",
    max_examples=10000000
)



