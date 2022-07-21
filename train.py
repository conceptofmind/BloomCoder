from transformers import BloomTokenizerFast, BloomForCausalLM
import torch

def BloomCoder():
    
    
    
    tokenizer = BloomTokenizerFast.from_pretrained("bigscience/bloom-1b3")
    model = BloomForCausalLM.from_pretrained("bigscience/bloom-1b3")

