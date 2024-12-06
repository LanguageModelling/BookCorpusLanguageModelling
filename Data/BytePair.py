from .utils import Indexer
from collections import Counter
import os.path
import pickle as pkl
import torch

class BytePairEncoder():
    def __init__(self):
        self.indexer = Indexer()
    def add_characters(self):
        with open('Data/BPEs/corpus_chars.txt', 'r') as f:
            [self.indexer.add_and_get_index(x.strip('\n')) for x in f]
    def _encode(self, sentence):
        encoding = [self.indexer.index_of(char) for char in sentence]
        changed = True
        while changed:
            changed = False
            new_encoding = []
            idx = 0
            while idx < len(encoding)-1:
                bigram = self.indexer.get_object(encoding[idx]) + self.indexer.get_object(encoding[idx+1])
                if self.indexer.contains(bigram):
                    new_encoding.append(self.indexer.index_of(bigram))
                    idx += 2
                    changed = True
                else:
                    new_encoding.append(encoding[idx])
                    idx += 1
            new_encoding.append(encoding[-1])
            encoding = new_encoding
        return encoding

    def encode(self, sentence):
        return torch.LongTensor(self._encode(sentence))
    
    def decode(self, sentence):
        output = ""
        for x in sentence:
            output += self.indexer.get_object(x.item())
        return output

    def bigramify(self, split_sentence):
        '''
        :param: split_sentence: List(str) : sentence encoding into tokens by current encoding method
        :return: new_corpus : encoding corpus bigrams
        '''
        special_characters = [self.indexer.index_of(x) for x in ['[', ']']]

        bigrams = [(split_sentence[i], split_sentence[i+1]) for i in range(len(split_sentence)-1) 
                   if split_sentence[i] not in special_characters
                     and split_sentence[i+1] not in special_characters]
        return bigrams


    def train(self, dataset, vocab_size=1000):
        path = f'Data/BPEs/{vocab_size}.pkl'
        if os.path.isfile(path):
            self.indexer.objs_to_ints, self.indexer.ints_to_objs = pkl.load(open(path, "rb"))
        else:
            # Flatten data
            encoding_data = ''
            for idx, data in enumerate(dataset):
                encoding_data += preprocess(data['text'])
            corpus = encoding_data
            # Create vocab]
            self.add_characters()
            while len(self.indexer) < vocab_size:
                bigram_counts = Counter()
                encoding = self._encode(corpus)
                bigrams = self.bigramify(encoding)
                bigram_counts.update(bigrams)
                new_word_i, new_word_i_plus_one = bigram_counts.most_common(1)[0][0]
                new_word = self.indexer.get_object(new_word_i) + self.indexer.get_object(new_word_i_plus_one)
                self.indexer.add_and_get_index(new_word)
                print(f'{len(self.indexer)}:{new_word}')
            # Save
            dicts = [self.indexer.objs_to_ints, self.indexer.ints_to_objs]
            pkl.dump(dicts, open(path, "wb" ) )
        self.max_token_value = len(self.indexer)
