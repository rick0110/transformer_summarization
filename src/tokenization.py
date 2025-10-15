import os
import csv
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import BPE, WordPiece, Unigram
from tokenizers.trainers import BpeTrainer, WordPieceTrainer, UnigramTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.normalizers import Sequence, NFD, Lowercase, StripAccents
import torch
import torch.nn as nn
import math

def train_and_save_tokenizer(
    dataset_path: str,
    text_columns: list[str],
    output_path: str,
    vocab_size: int,
    model_type: str = 'wordpiece',
    special_tokens: list[str] | None = None,
    verbose: bool = True,
    max_examples: int | None = None,  # <- novo parâmetro
):
    if special_tokens is None:
        special_tokens = ["[UNK]", "[PAD]", "[SOS]", "[EOS]"]

    if not text_columns:
        raise ValueError("text_columns não pode ser vazio. Passe a lista de colunas de texto do dataset.")

    if verbose:
        print(f"Iniciando o treinamento do tokenizer com o modelo '{model_type}'...")
        print(f"Carregando dataset de '{dataset_path}' (carregando todos os dados em memória)...")

    # carregar todo o dataset na memória (modo simples)
    ds = load_dataset('json', data_files={'train': dataset_path})
    dataset = ds['train']

    model_type_l = model_type.lower()
    if model_type_l == 'wordpiece':
        model = WordPiece(unk_token="[UNK]")
        trainer = WordPieceTrainer(vocab_size=vocab_size, special_tokens=special_tokens)
    elif model_type_l == 'bpe':
        model = BPE(unk_token="[UNK]")
        trainer = BpeTrainer(vocab_size=vocab_size, special_tokens=special_tokens)
    elif model_type_l == 'unigram':
        model = Unigram()
        trainer = UnigramTrainer(vocab_size=vocab_size, unk_token="[UNK]", special_tokens=special_tokens)
    else:
        raise ValueError("model_type deve ser 'wordpiece', 'bpe' ou 'unigram'")

    tokenizer = Tokenizer(model)
    # normalização simples: decompor, lowercase, remover acentos
    tokenizer.normalizer = Sequence([NFD(), Lowercase(), StripAccents()])
    tokenizer.pre_tokenizer = Whitespace()

    # construir lista simples de textos para treino
    training_texts: list[str] = []
    count = 0
    for example in dataset:
        if isinstance(example, dict):
            for col in text_columns:
                text = example.get(col)
                if text:
                    training_texts.append(str(text))
        else:
            for col in text_columns:
                text = getattr(example, col, None)
                if text:
                    training_texts.append(str(text))
        count += 1
        if max_examples is not None and count >= max_examples:
            break

    if verbose:
        print(f"Treinando o tokenizer com um vocabulário de {vocab_size} tokens. Isso pode levar um tempo...")

    try:
        # passar a lista inteira (ou um iterador) para o trainer
        tokenizer.train_from_iterator(iter(training_texts), trainer)
    except Exception as e:
        raise RuntimeError(f"Falha ao treinar o tokenizer: {e}") from e

    if verbose:
        print("Treinamento concluído.")
        print(f"Vocab size solicitado: {vocab_size}")
        try:
            print(f"Vocab size real gerado: {tokenizer.get_vocab_size()}")
        except Exception:
            pass

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    tokenizer.save(output_path)
    if verbose:
        print(f"Tokenizer salvo com sucesso em: '{output_path}'")

    # mostrar amostra de tokens e tokenização de exemplo
    if verbose:
        try:
            vocab = tokenizer.get_vocab()
            inv = sorted(vocab.items(), key=lambda x: x[1])
            sample_tokens = [t for t, _ in inv[:50]]
            print("Amostra dos primeiros tokens:", sample_tokens)
        except Exception:
            pass

        try:
            examples = training_texts[:3]
            for s in examples:
                enc = tokenizer.encode(s)
                print(f"Texto: {s}")
                print(f"Tokens: {enc.tokens}")
        except Exception:
            pass

    return tokenizer

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1)]
        return x

class TransformerEmbeddings(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, max_len: int, padding_idx: int, dropout_rate: float):
        super().__init__()
        self.token_embeddings = nn.Embedding(vocab_size, d_model, padding_idx=padding_idx)
        self.positional_embeddings = PositionalEncoding(d_model, max_len)
        self.dropout = nn.Dropout(dropout_rate)
        self.d_model = d_model

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        token_embeds = self.token_embeddings(input_ids) * math.sqrt(self.d_model)
        final_embeds = self.positional_embeddings(token_embeds)
        return self.dropout(final_embeds)
    
