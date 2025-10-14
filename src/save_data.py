import json
from datasets import load_dataset
from tqdm import tqdm
from tokenization import train_and_save_tokenizer

dataset = load_dataset('Anthropic/hh-rlhf', split='train')

output_file = "./../data/hh-rlhf.json"
with open(output_file, 'w', encoding='utf-8') as f:
    for item in tqdm(dataset):
        f.write(json.dumps(item, ensure_ascii=False) + '\n')

tokenizer_output_path = "./../data/hh-rlhf_tokenizer.json"
# ...existing code...
tokennn = train_and_save_tokenizer(
    dataset_path=output_file,
    text_columns=["problem", "generated_solution"],
    output_path=tokenizer_output_path,
    vocab_size=500,
    model_type="wordpiece",
    max_examples=1_000_000  
)



