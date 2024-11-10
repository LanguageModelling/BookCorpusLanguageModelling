#DataLoader.py

import pandas as pd
from torch.utils.data import Dataset
import random
import pyarrow.parquet as pq #reading Parquet files in chunks

class BookCorpusDataset(Dataset):
    def __init__(self, file_path, chunk_size=1000, max_samples=100,random_sample=False):
        self.file_path = file_path
        self.chunk_size = chunk_size
        self.max_samples = max_samples # max samples for testing/debugging
        self.random_sample = random_sample # not using, optional
        self.sample_count = 0  # keep track how many samples have been processed so far.
        self.total_examples = 74004228  #total number of examples in bookcorpus.parquet, hardcoded here
        self.chunk_iterator = self._read_parquet_in_chunks() # read the file in batches of self.chunk_size (e.g., 1000 rows)
    
    def _read_parquet_in_chunks(self):
        # Open Parquet file using pyarrow to stream it
        parquet_file = pq.ParquetFile(self.file_path)
        for batch in parquet_file.iter_batches(batch_size=self.chunk_size):
            yield pd.DataFrame.from_records(batch.to_pandas().to_dict(orient='records'))

    def __len__(self):
        return self.max_samples  # Only process max_samples for debugging or testing

    def __getitem__(self, idx):
        
        if self.sample_count >= self.max_samples: #checks the number of samples processed so far
            raise IndexError("Exceeded the max_samples limit")

        # Fetch the next chunk
        if self.sample_count == 0:
            self.chunk_data = next(self.chunk_iterator)  # Get the first chunk

        sample = self.chunk_data.iloc[idx]  # Fetch the row from the chunk
        self.sample_count += 1
        return sample


##########Test DataLoader.py 
from DataLoader import BookCorpusDataset

if __name__ == "__main__":
    # Set max_samples to 100 for quick testing/debugging
    dataset = BookCorpusDataset("bookcorpus.parquet", chunk_size=1000, max_samples=100)

    # Fetch and print a few samples to test
    for i in range(len(dataset)):  # Iterate through only a subset (100 examples)
        print(dataset[i])
        
##########Output
# Name: 0, dtype: object
#text    but just one look at a minion sent him practic...
#Name: 1, dtype: object
##text    that had been megan 's plan when she got him d...
#Name: 2, dtype: object
#text    he 'd seen the movie almost by mistake , consi...
#Name: 3, dtype: object
#text    she liked to think being surrounded by adults ...
