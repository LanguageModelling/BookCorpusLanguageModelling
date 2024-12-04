from typing import Any
import numpy as np
import datasets
from .Preprocessing import preprocess
from .IterableDataset import BookCorpusDataset
from .BytePair import BytePairEncoder
import torch
import torch.nn.functional as F
from tiktoken import get_encoding
import time
import os


class Library:
    def __init__(self, encoding = 1000, train_size = 2**20, test_size = 2**16, streaming=True):
        self.streaming=streaming
        self.train_size = train_size
        self.test_size = test_size
        if streaming:
            self.dataset = datasets.load_dataset('bookcorpus', streaming=streaming, trust_remote_code=True, split=f'train[:{self.test_size+self.train_size}]')
        else:
            path = f'Data/bookcorpus.pt'
            if os.path.isfile(path):
                self.dataset = torch.load(path, weights_only=False)
            else:
                self.dataset = datasets.load_dataset('bookcorpus', streaming=streaming, trust_remote_code=True)[f'train[:{self.texts_size+self.train_size}']
                torch.save(self.dataset, path)

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
                    if idx < self.train_size:
                        encoding_data += preprocess(data['text'])
                    else:
                        break
                self.encoding.train(encoding_data, vocab_size=encoding)
        self.train_generator = self._create_train_generator()
        self.test_generator = self._create_test_generator()
                
    
    def _create_train_generator(self):
        for idx, data in enumerate(self.dataset):
            if self.train_size+self.test_size > idx >= self.test_size:
                for token in self.encoding.encode(preprocess(data['text'])):
                    yield token
            elif idx > self.train_size+self.test_size:
                return
    
    def _create_test_generator(self):
        for idx, data in enumerate(self.dataset):
            if idx < self.test_size:
                for token in self.encoding.encode(preprocess(data['text'])):
                    yield token
            else:
                return
            
    def get_train_dataloader(self, batch_size):
        dataset = BookCorpusDataset(self._create_train_generator())
        dataloader = torch.utils.data.DataLoader(dataset, batch_size, shuffle=False)
        return dataloader
    
    def get_test_dataloader(self, batch_size):
        return torch.utils.data.DataLoader(BookCorpusDataset(self._create_test_generator()), batch_size, shuffle=False)
    
    def ngramify(batch_indices, n=2):
        batch_size, seq_length = batch_indices.shape
        new_indices = torch.zeros([batch_size, seq_length-n, n])
        for nx in range(n):
            new_indices[:, :, n] = batch_indices[:,nx:nx+seq_length-n, nx]
        return new_indices
    
    def calc_perplexity(self, model, seq_length = 512, batch_size = 128, ngram=False):
        test_dataloader = self.get_test_dataloader(seq_length)
        log_prob = 0.0
        x_batch = torch.zeros([batch_size, seq_length-1])
        y_batch = torch.zeros([batch_size, seq_length-1])
        for idx, data in enumerate(test_dataloader):
            x_batch[idx] = data[:-1]
            y_batch[idx] = data[1:]
            if idx == batch_size-1:
                # Run model
                if ngram:
                    x_batch = self.ngramify(x_batch)
                y_pred = model(x_batch.long()).detach()
                y_map = F.one_hot(y_batch.long(), num_classes=model.vocab_size).mT
                log_probs = torch.sum(y_pred*y_map)/(seq_length)
                return np.exp(-log_probs/idx)

    def shannon(self, model, length=100):
        current_sentence = torch.LongTensor(self.encoder.encode('[')).unsqueeze(0)
        for i in range(length):
            output = torch.exp(model(current_sentence))[0,:,-1]
            new_char=torch.distributions.categorical.Categorical(probs=output).sample()
            current_sentence = torch.cat((current_sentence, new_char.unsqueeze(0).unsqueeze(0)))
        return self.encoder.decode(current_sentence)
    
    # Code from https://discuss.pytorch.org/t/how-do-i-check-the-number-of-parameters-of-a-model/4325/5
    def get_n_params(self, model):
        pp=0
        for p in list(model.parameters()):
            nn=1
            for s in list(p.size()):
                nn = nn*s
            pp += nn
        return pp