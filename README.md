# Bookcorpus LanguageModeling
For our project, we are contrasting a set of neural language models with
our metric being perplexity.

## Table of Contents

- [Project Overview](#project-overview)
- [Implemented Models](#implemented-models)
  - [MLP](#mlp)
  - [RNN](#rnn)
  - [S4](#s4)
  - [Transformer](#transformer)
- [Dataset Preparation](#dataset-preparation)
- [Training and Evaluation](#training-and-evaluation)
- [Usage](#usage)

---

## Project Overview

For our project, we implemented and evaluated
three language models to understand their effective-
ness in sequence modeling tasks. Language model-
ing is a cornerstone of natural language processing,
and selecting the right model can significantly im-
pact efficiency and performance. By comparing
these models with different vocab sizes and struc-
tural complexities, we aim to provide insights into
their strengths and limitations. 
---

## Implemented Models

### MLP(N-gram)
- **Description**: A feedforward neural network that processes sequences by flattening them into a single vector. Suitable for tasks where temporal dependencies are not critical.
- **Key Components**:
  - Fully connected layers with ReLU activations.
  - Dropout regularization.
  - Does not explicitly model temporal dependencies.

---

### RNN
- **Description**: A vanilla recurrent neural network that models temporal dependencies through recurrent connections.
- **Key Components**:
  - Token embeddings.
  - Fully connected recurrent layers with non-linear activations (ReLU).
  - Output layer projecting to the vocabulary size.
  - LogSoftmax for output probabilities.

---

### S4 (State Space Models)
- **Description**: Implements a state-space model for efficient and expressive sequence modeling, focusing on capturing long-term dependencies.
- **Key Components**:
   - Implements the state-space mechanism to process sequential data efficiently.
   - Utilizes discretized Legendre polynomials for handling long-term dependencies.
   - Provides both **convolutional** (`get_legendre_conv`) and **recurrent** (`get_legendre_rec`) processing modes.

---

### Transformer
- **Description**: A GPT-like Transformer model that uses self-attention for parallel processing of sequences.
- **Key Components**:
  - Token and positional embeddings.
  - Multi-head self-attention with causal masking.
  - Feedforward layers with residual connections and LayerNorm.
  - LogSoftmax for output probabilities.

---

## Dataset Preparation
Dataset link: https://huggingface.co/datasets/bookcorpus/bookcorpus

The project uses a custom `Library` class to generate and manage datasets:
- **Encoding**: Tokens are encoded using a defined `encoding` size.
- **Train/Test Split**: The dataset is divided into training and test sets.
- **DataLoader**: Provides sequential data in batches for training and evaluation.

---

## Training and Evaluation
We compared the above three models in a series
of sizes.

For all models, Adam optimizer was used with a
negative log-likelihood loss function. Models were
trained up to 64 epochs. The set of nine models
were trained on the Discovery cluster at Northeast-
ern, using 16GB of RAM, and using either a T4,
V100, or A100 GPU. 

For evaluation, we used the same starting conditions and the iteration count that minimized the perplexity.

---

## Usage

### Requirements
- Python 3.8+
- PyTorch 2.0+
- GPU with CUDA support (recommended)
