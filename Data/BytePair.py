from utils import Indexer
from collections import Counter

class BytePairEncoder():
    def __init__(self):
        self.indexer = Indexer()
    
    def add_characters(self, corpus):
        '''
        :param corpus: List of strings representing every sentence in the corpus
        :return None:
        '''

        # Split all sentences on a character level
        for idx, sentence in enumerate(corpus):
            [self.indexer.add_and_get_index(char) for char in sentence]

    def encode(self, sentence):
        encoding = list(sentence)
        changed = True
        while changed:
            changed = False
            new_encoding = []
            idx = 0
            while idx < len(encoding)-1:
                if encoding[idx] + encoding[idx+1] in self.indexer.objs_to_ints.keys():
                    new_encoding.append(encoding[idx] + encoding[idx+1])
                    idx += 2
                    changed = True
                else:
                    new_encoding.append(encoding[idx])
                    idx += 1
            new_encoding.append(encoding[-1])
            encoding = new_encoding
        return encoding
                

    def bigramify(self, split_sentence):
        '''
        :param: split_sentence: List(str) : sentence encoding into tokens by current encoding method
        :return: new_corpus : encoding corpus bigrams
        '''
        bigrams = [split_sentence[i] + split_sentence[i+1] for i in range(len(split_sentence)-1)]
        return bigrams


    def train(self, corpus, vocab_size=1000, char_level = False):
        # Create vocab
        self.add_characters(corpus)
        while len(self.indexer) < vocab_size and not char_level:
            bigram_counts = Counter()
            for sentence in corpus:
                encoding = self.encode(sentence)
                bigrams = self.bigramify(encoding)
                bigram_counts.update(bigrams)
            new_word = bigram_counts.most_common(1)[0][0]
            if new_word in self.indexer.objs_to_ints.keys():
                assert False
            self.indexer.add_and_get_index(new_word)
datalength = 1000000
from Library import Library
import time
import matplotlib.pyplot as plt
lib = Library()
dataset = lib.get_examples(datalength)
times = []
for vocab_size in [100]:
    bpe = BytePairEncoder()
    tic = time.time()
    bpe.train(dataset, vocab_size)
    times.append(time.time()-tic)
    print(times[-1])

plt.plot(times)
plt.show()
    