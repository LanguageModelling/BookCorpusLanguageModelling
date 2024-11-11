from typing import Any
import datasets
from Preprocessing import preprocess
from IterableDataset import BookCorpusDataset
import torch
from tiktoken import get_encoding
class Library:
    def __init__(self, encoding = None, test_size = 2**16):
        self.test_size = test_size
        self.dataset = datasets.load_dataset('bookcorpus', streaming=True, trust_remote_code=True)['train']
        self.train_generator = self._create_train_generator()
        self.test_generator = self._create_test_generator()
        match encoding:
            case '200k':
                self.encoding = get_encoding('o200k_base')
            case '100k':
                self.encoding = get_encoding('cl100k_base')
            case _:
                self.encoding = get_encoding('r50k_base')
                
    
    def _create_train_generator(self):
        for idx, data in enumerate(self.dataset):
             if idx >= self.test_size:
                for token in self.encoding.encode(preprocess(data['text'])):
                    yield token
    
    def _create_test_generator(self):
        for idx, data in enumerate(self.dataset):
            if idx < self.test_size:
                for token in self.encoding.encode(preprocess(data['text'])):
                    yield token
    
    def get_train_dataloader(self, batch_size):
        return torch.utils.data.DataLoader(BookCorpusDataset(self._create_train_generator()), batch_size, shuffle=False)
    
    def get_test_dataloader(self, batch_size):
        return torch.utils.data.DataLoader(BookCorpusDataset(self._create_test_generator()), batch_size, shuffle=False)
if __name__ == '__main__':
    lib = Library()
    dataloader = lib.get_train_dataloader(64)
    test_dataloader = lib.get_test_dataloader(64)
