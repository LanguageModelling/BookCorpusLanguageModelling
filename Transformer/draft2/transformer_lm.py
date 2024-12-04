# models.py

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from lm import *
import random


class LanguageModel(object):

    def get_next_char_log_probs(self, context) -> np.ndarray:
        """
        Returns a log probability distribution over the next characters given a context.
        The log should be base e

        NOTE: You should make sure you call model.eval() to determinize inference here (turns off dropout
        layers in TransformerEncoder).
        :param context: the string context that the LM conditions on
        :return: A numpy vector log P(y | context) where y ranges over the output vocabulary.
        """
        raise Exception("Only implemented in subclasses")


    def get_log_prob_sequence(self, next_chars, context) -> float:
        """
        Scores a bunch of characters following context. That is, returns
        log P(nc1, nc2, nc3, ... | context) = log P(nc1 | context) + log P(nc2 | context, nc1), ...
        The log should be base e

        NOTE: You should make sure you call model.eval() to determinize inference here (turns off dropout
        layers in TransformerEncoder).
        :param next_chars:
        :param context:
        :return: The float probability
        """
        raise Exception("Only implemented in subclasses")


class UniformLanguageModel(LanguageModel):
    def __init__(self, voc_size):
        self.voc_size = voc_size

    def get_next_char_log_probs(self, context):
        return np.ones([self.voc_size]) * np.log(1.0/self.voc_size)

    def get_log_prob_sequence(self, next_chars, context):
        return np.log(1.0/self.voc_size) * len(next_chars)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, num_positions: int = 20, batched=False):
        """
        :param d_model: dimensionality of the embedding layer to your model; since the position encodings are being
        added to character encodings, these need to match (and will match the dimension of the subsequent Transformer
        layer inputs/outputs)
        :param num_positions: the number of positions that need to be encoded; the maximum sequence length this
        module will see
        :param batched: True if you are using batching, False otherwise
        """
        super().__init__()
        # Dict size
        self.emb = nn.Embedding(num_positions, d_model)
        self.batched = batched
        

    def forward(self, x):
        """
        :param x: If using batching, should be [batch size, seq len, embedding dim]. Otherwise, [seq len, embedding dim]
        :return: a tensor of the same size with positional embeddings added in
        """
        # Second-to-last dimension will always be sequence length
        input_size = x.shape[-2]
        indices_to_embed = torch.tensor(np.asarray(range(0, input_size))).type(torch.LongTensor)
        if self.batched:
            # Use unsqueeze to form a [1, seq len, embedding dim] tensor -- broadcasting will ensure that this
            # gets added correctly across the batch
            emb_unsq = self.emb(indices_to_embed).unsqueeze(0)
            return x + emb_unsq
        else:
            return x + self.emb(indices_to_embed)



class NeuralLanguageModel(LanguageModel, nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_layers, dim_feedforward, max_seq_length,vocab_index):
        nn.Module.__init__(self)
        LanguageModel.__init__(self)
        self.model = TransformerLM(vocab_size, d_model, nhead, num_layers, dim_feedforward, max_seq_length)
        self.vocab_index  = vocab_index
        

    def get_next_char_log_probs(self, context):
        context = " " + context
        self.model.eval()
        with torch.no_grad():
            context_tensor = torch.LongTensor([self.vocab_index.index_of(c) for c in context])
            log_probs = self.model(context_tensor)[-1]

            return log_probs.numpy()
        

    def get_log_prob_sequence(self, next_chars, context):
        context = " " + context
        self.model.eval()
        with torch.no_grad():
            
            context_tensor = torch.LongTensor([self.vocab_index.index_of(c) for c in context])
            next_chars_tensor = torch.LongTensor([self.vocab_index.index_of(c) for c in next_chars])
            input_tensor = torch.cat((context_tensor, next_chars_tensor), dim = 0)
            log_probs = self.model(input_tensor)
            next_char_probs = log_probs[(-len(next_chars)-1):-1]
            next_sequence_mask = torch.nn.functional.one_hot(next_chars_tensor, 27)
            log_probs_sequence = torch.sum(next_sequence_mask*next_char_probs)
            
            # log_probs_sequence = sum(log_probs[len(context)+i, next_chars_tensor[0,i]].item() for i in range(len(next_chars)))
            return float(log_probs_sequence)
            
        

class TransformerLM(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_layers, dim_feedforward, max_seq_length):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length, batched=False)
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers,num_layers)
        self.fc_out = nn.Linear(d_model, vocab_size)
        self.log_softmax = nn.LogSoftmax(dim = -1)
    def forward(self, src):
        # print('shape of input ',src.shape)
        src = self.embedding(src)
        # print('shape of embedding output', src.shape)
        src = self.positional_encoding(src).unsqueeze(1)
        # print(" positional encoding shape ",src.shape)
        
        src_mask = self.generateMask(src.shape[0])
        src = self.transformer_encoder(src,src_mask).squeeze(1)
        # print('shape of output of transformer encoder ',src.shape)
        src = self.fc_out(src)
        # print('shape of first linear layer ',src.shape)
        src = self.log_softmax(src)
        
        print('shape of output(softmax) ', src.shape)
        # assert False
       
              
        return src
    def generateMask(self,n):
        mask = (torch.triu(torch.ones(n, n),1) == 1)
        mask = mask.float().masked_fill(mask == 1, float('-inf')).masked_fill(mask == 0, float(0.0))
        # print(mask)
        # assert False
        return mask
        



        
    
    

def train_lm(args, train_text, dev_text, vocab_index):
    """
    :param args: command-line args, passed through here for your convenience
    :param train_text: train text as a sequence of characters
    :param dev_text: dev text as a sequence of characters
    :param vocab_index: an Indexer of the character vocabulary (27 characters)
    :return: a NeuralLanguageModel instance trained on the given data
    """
    
    vocab_size = len(vocab_index)
    d_model = 64 #needs to high
    nhead = 1
    num_layers = 4
    dim_feedforward = 256 #needs to be very high
    max_seq_length  = 1000
    model = NeuralLanguageModel(vocab_size, d_model, nhead, num_layers, dim_feedforward, max_seq_length,vocab_index)
    model.train()
    optimizer = optim.Adam(model.parameters(), lr = 0.001)
    random.seed(43678)
    criterion = nn.NLLLoss()

    data_length = len(train_text)

    num_epochs = 75
    chunk_size = 500
    print(len(train_text)," lenght of training examples")
    for epoch in range(num_epochs):
        start_index = random.randint(0,data_length-1)
        end_index = start_index+data_length
        
        
        total_loss = 0
        for i in range(start_index,end_index, chunk_size):
            context_chunk = train_text[i%data_length:(i%data_length)+chunk_size]
            target_chunk = train_text[(i+1)%data_length:((i+1)%data_length)+chunk_size]
            if len(context_chunk) < chunk_size or len(target_chunk) < chunk_size:
                continue
            context_tensor = torch.LongTensor([vocab_index.index_of(c) for c in context_chunk])
            target_tensor = torch.LongTensor([vocab_index.index_of(c) for c in target_chunk])

            optimizer.zero_grad()
            output = model.model(context_tensor)
            loss = criterion(output.view(-1, vocab_size), target_tensor.view(-1))
            loss.backward()
            optimizer.step()
            total_loss+=loss
            
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {total_loss / ((len(train_text) - chunk_size) // chunk_size)}')
        print_evaluation(dev_text, model, vocab_index, 'output.json')
        
    return model
            
            
        
