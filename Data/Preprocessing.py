#Preprocessing.py

import re
import nltk
#import spacy #handles puntuation and contractions, advanced tokenization
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
#from transformers import BertTokenizer, GPT2Tokenizer

nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)

#initialize spaCy and NLTK
#nlp = spacy.load("en_core_web_sm")  # Load spaCy model for English
stop_words = set(stopwords.words("english"))
ps = PorterStemmer()

# initialize transformers tokenizers for BERT and GPT-2
#bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
#gpt2_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

def preprocess(text_input):
    """
    Preprocesses the input text by applying tokenization, lowercasing, removing stopwords,
    stemming, and adding special tokens.
    
    :param text_input: str: Input sentence to preprocess
    :return: str: Preprocessed sentence
    """
    
    
    # 2. Convert to lowercase
    text_input = text_input.lower()
    
    # 3. Remove uncommon characters (non-alphanumeric characters except spaces)
    text_input = re.sub(r"[^a-zA-Z0-9\s]", "", text_input) # Keeps letters, digits, and spaces

    # 4. Tokenization (using NLTK by default)
    # - Use NLTK's word_tokenize function for tokenization
    #words = nltk.word_tokenize(text_input)  #better than split()

    # optional: Using spaCy tokenization for comparison
    #spacy_tokens = [token.text for token in nlp(text_input) if not token.is_stop]
    
    # optional: Using BERT tokenizer
    #bert_tokens = bert_tokenizer.tokenize(text_input)
    
    # optional: Using GPT-2 tokenizer
    #gpt2_tokens = gpt2_tokenizer.tokenize(text_input)

    # Use NLTK tokenization by default
    #tokens = words   

    # 5. Removing stopwords (using NLTK's stopwords list)
    #tokens = [token for token in tokens if token not in stop_words]

    # 6. Stemming (using NLTK's Porter Stemmer)
    #tokens = [ps.stem(token) for token in tokens]

    # 7. Reconstructing the sentence
    #preprocessed_text = " ".join(tokens)
    preprocessed_text = text_input
    
    # 8. Remove extra spaces (if any)
    preprocessed_text = re.sub(r'\s+', ' ', preprocessed_text)
    
    # 9. Strip leading/trailing whitespaces
    preprocessed_text = preprocessed_text.strip()

    # 1. Add start and end tokens 
    preprocessed_text = "[START] " + preprocessed_text + " [END]"

    return preprocessed_text

if __name__ == "__main__":
    ######## Test Preprocessing.py
    text_input = "but just one look at a minion sent him practic"
    processed_text = preprocess(text_input)
    print(processed_text)
    #####Output using spacy
    #  <start> look minion sent practic <end>

    #####Output using NLTK tokenization 
    #  start one look minion sent practic end
