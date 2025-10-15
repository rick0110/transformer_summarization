import json
from datasets import load_dataset
from tqdm import tqdm
from tokenization import train_and_save_tokenizer

dataset = load_dataset('HuggingFaceTB/cosmopedia-100k', split='train')

output_file = "./../data/HuggingFaceTB.json"
with open(output_file, 'w', encoding='utf-8') as f:
    for item in tqdm(dataset):
        f.write(json.dumps(item, ensure_ascii=False) + '\n')

tokenizer_output_path = "./../data/HuggingFaceTB_tokenizer.json"
# ...existing code...
tokennn = train_and_save_tokenizer(
    dataset_path=output_file,
    text_columns=["problem", "generated_solution"],
    output_path=tokenizer_output_path,
    vocab_size=2000,
    model_type="bce",
    max_examples=1_000_000  
)



