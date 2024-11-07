import datasets
from Preprocessing import preprocess
class Library:
    def __init__(self):
        self.dataset = datasets.load_dataset('bookcorpus', streaming=True, trust_remote_code=True)['train']
        self.preprocessed_dataset = self.dataset.map(preprocess)
        self.generator = self._create_generator()
    
    def _create_generator(self):
        for data in self.preprocessed_dataset:
            yield data['text']
    
    def get_example(self):
        return next(self.generator)

lib = Library()
for i in range(10):
    print(lib.get_example())