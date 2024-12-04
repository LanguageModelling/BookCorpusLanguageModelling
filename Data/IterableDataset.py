#DataLoader.py

import pandas as pd
from torch.utils.data import IterableDataset

class BookCorpusDataset(IterableDataset):
    def __init__(self, generator):
        self.generator = generator
    
    def __iter__(self):
        return self.generator
    
    def __len__(self):
        return sum(1 for _ in self.generator) # Add 1 to sum for each index in generator