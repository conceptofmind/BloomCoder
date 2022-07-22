import copy
from itertools import chain
from datasets import load_dataset
from torch.utils.data import DataLoader, DistributedSampler
from torch.distributed import get_world_size
from transformers import BloomTokenizerFast, default_data_collator
from .hugging_face_config import CFG

def build_dataloaders(cfg: CFG):
    """
    Build dataloaders for the Bloom Coder model.
    """

    # Load training dataset
    load_train_data = load_dataset(cfg.train_dataset_name, split = cfg.choose_train_split)

    # Remove unused columns from the training dataset
    load_train_data = load_train_data.remove_columns(cfg.remove_train_columns)

    # Load validation dataset
    load_eval_data = load_dataset(cfg.eval_dataset_name, split = cfg.choose_eval_split)

    # Remove unused columns from the validation dataset
    load_eval_data = load_eval_data.remove_columns(cfg.remove_eval_columns)

    # Shuffle the training input files.
    shuffled_train_files = load_train_data.shuffle(seed = cfg.seed)

    # Shuffle the validation input files.
    shuffled_eval_files = load_eval_data.shuffle(seed = cfg.seed)

    tokenizer = BloomTokenizerFast.from_pretrained(cfg.tokenizer_name)

    """
    A sequence length of x is used for the model. Input examples are concatenated
    together and then split into sequences of exactly x tokens, so that there are 
    no padding tokens, but examples may be split in the middle.

    Tokenize function reference:
    https://github.com/hpcaitech/PaLM-colossalai/blob/main/data/wikitext.py
    """

    def tokenize(examples):
        seq_length = cfg.tokenizer_seq_length
        examples = tokenizer(examples[cfg.select_input_string])
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        if total_length >= seq_length:
            total_length = (total_length // seq_length) * seq_length

        result = {
            k: [t[i : i + seq_length] for i in range(0, total_length, seq_length)]
            for k, t in concatenated_examples.items()
        }

        result["labels"] = copy.deepcopy(result["input_ids"])

        return result
    
    """
    Map the tokenization function to the shuffled training files to create an 
    Iterable training dataset of batched input sequences of x tokens.
    Remove columns from the the shuffled training files so that you are left with 
    only the input_ids, attention_mask, and labels columns.
    """
    
    tokenized_train_dataset = shuffled_train_files.map(tokenize, batched = True, remove_columns = [cfg.select_input_string])

    """
    Map the tokenization function to the shuffled validation files to create an 
    Iterable validation dataset of batched input sequences of x tokens.
    Remove columns from the the shuffled training files so that you are left with 
    only the input_ids, attention_mask, and labels columns.
    """
    
    tokenized_eval_dataset = shuffled_eval_files.map(tokenize, batched = True, remove_columns = [cfg.select_input_string])

    # Convert the format of the tokenized train dataset to PyTorch Tensors
    train_with_torch = tokenized_train_dataset.set_format(type = "torch")

    # Convert the format of the tokenized validation dataset to PyTorch Tensors
    eval_with_torch = tokenized_eval_dataset.set_format(type = "torch")

    # Train dataset used for sampling.
    sample_train_dataset = DistributedSampler(train_with_torch, shuffle = True) if get_world_size() > 1 else None

    # Validation dataset used for sampling.
    sample_eval_dataset = DistributedSampler(eval_with_torch, shuffle = False) if get_world_size() > 1 else None

    # Create the train dataloader. If the length of a tokenized input sequence is less than 2048 drop it.
    train_dataloader = DataLoader(tokenized_train_dataset, shuffle = True, sampler = sample_train_dataset, drop_last = True, collate_fn = default_data_collator, batch_size = cfg.batch_size)

    # Create the validation dataloader. If the length of a tokenized input sequence is less than 2048 drop it.
    eval_dataloader = DataLoader(tokenized_eval_dataset, sampler = sample_eval_dataset, drop_last = True, collate_fn = default_data_collator, batch_size = cfg.batch_size)

    # Return the training and validation dataloaders to be used in the model
    print('Done building dataloaders')
    return train_dataloader, eval_dataloader

if __name__ == '__main__':
    
    # Get Dataloader Configuration Arguments
    data_loader_args = CFG()

    # Test Build Dataloaders
    train_loader, eval_loader = build_dataloaders(cfg = data_loader_args)

    print(next(iter(train_loader))['input_ids'])
    print(next(iter(train_loader))['input_ids'].shape)