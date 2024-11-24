from typing import Any
import datasets
from Preprocessing import preprocess
from IterableDataset import BookCorpusDataset
from BytePair import BytePairEncoder
import torch
import torch.nn.functional as F
from tiktoken import get_encoding
import time


class Library:
    def __init__(self, encoding = 1000, train_size = 2**20, test_size = 2**12):
        self.train_size = train_size
        self.test_size = test_size
        self.dataset = datasets.load_dataset('bookcorpus', streaming=True, trust_remote_code=True)['train']
        match encoding:
            case '200k':
                self.encoding = get_encoding('o200k_base')
            case '100k':
                self.encoding = get_encoding('cl100k_base')
            case '50k':
                self.encoding = get_encoding('r50k_base')
            case _ :
                # Byte Pair
                self.encoding = BytePairEncoder()
                encoding_data = ''
                for idx, data in enumerate(self.dataset):
                    if idx < encoding:
                        encoding_data += preprocess(data['text'])
                    else:
                        break
                self.encoding.train(encoding_data, vocab_size=encoding)

        self.train_generator = self._create_train_generator()
        self.test_generator = self._create_test_generator()
                
    
    def _create_train_generator(self):
        for idx, data in enumerate(self.dataset):
            if self.train_size > idx >= self.test_size:
                for token in self.encoding.encode(preprocess(data['text'])):
                    yield token
                    time.sleep(.0001)
        return
    
    def _create_test_generator(self):
        for idx, data in enumerate(self.dataset):
            if idx < self.test_size:
                for token in self.encoding.encode(preprocess(data['text'])):
                    yield token
                    time.sleep(.0001)
            else:
                return
            
    def get_train_dataloader(self, batch_size):
        return torch.utils.data.DataLoader(BookCorpusDataset(self._create_train_generator()), batch_size, shuffle=False)
    
    def get_test_dataloader(self, batch_size):
        return torch.utils.data.DataLoader(BookCorpusDataset(self._create_test_generator()), batch_size, shuffle=False)
    
    def calc_perplexity(self, model, seq_length = 512):
        test_dataloader = self.get_test_dataloader(seq_length+1)
        log_prob = 0.0
        for idx, data in enumerate(test_dataloader):
            y_pred = model(data[:-1].unsqueeze(0).long()).squeeze(0)
            y_map = F.one_hot(data[1:], num_classes=model.vocab_size).mT
            seq_probs = torch.sum(y_pred*y_map, dim=1)
            log_prob += torch.sum(seq_probs)/seq_length
        return float(torch.exp(-log_prob/idx).item())

if __name__ == '__main__':
    lib = Library()
    dataloader = lib.get_train_dataloader(64)
    test_dataloader = lib.get_test_dataloader(64)
