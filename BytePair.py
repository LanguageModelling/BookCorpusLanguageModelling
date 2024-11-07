from utils import Indexer
def text_preprocess(sentence):
    '''
    TODO: Fill in with text preprocessing steps
    '''
    sentence = sentence.lower()
    return sentence

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
        start_idx = 0
        end_idx = 0
        encoding = []
        while end_idx < len(sentence):
            if sentence[start_idx:end_idx] in self.indexer.objs_to_ints.keys():
                current_word = sentence[start_idx:end_idx]
                end_idx += 1
            else:
                encoding.append(self.indexer.index_of(current_word))
                start_idx = end_idx

    def train(self, corpus, vocab_size=1000, char_level = False):
        # Create vocab
        self.add_characters(corpus)

datalength = 1000
from bookcorpus import Bookcorpus
import time
corpus = Bookcorpus()
dataset = [corpus._generate_examples() for _ in range(datalength)]
bpe = BytePairEncoder()
tic = time.time()
bpe.train(dataset, 100)
print(time.time()-tic)

    