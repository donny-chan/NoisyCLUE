from time import time

from transformers import MT5Tokenizer, MT5ForConditionalGeneration
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, Dataset

class MT5FineTuner:
    def __init__(
        self, 
        model: MT5ForConditionalGeneration, 
        tokenizer: MT5Tokenizer, 
        **kwargs):
        self.model = model
        self.tokenizer = tokenizer
        self.args = kwargs
        
    def train(self):
        pass
    
    def training_step(self):
        '''Forward and (possibly) backward'''
        
    
    def predict(self, dataset):
        