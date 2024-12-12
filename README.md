# Bookcorpus LanguageModeling
For this project, we implemented and evaluated
three language models to understand their effective
ness in sequence modeling tasks. Language model
ing is a cornerstone of natural language processing,
and selecting the right model can significantly im
pact efficiency and performance. By comparing
these models with different vocab sizes and struc
tural complexities, we aim to provide insights into
their strengths and limitations.

## Table of Contents

- [Dataset Preparation](#dataset-preparation)
- [Implemented Models](#implemented-models)
  - [MLP](#mlp)
  - [RNN](#rnn)
  - [S4](#s4)
  - [Transformer](#transformer)
- [Training and Evaluation](#training-and-evaluation)
- [How to use the models](#how-to-use-the-models)
- [Usage](#usage)

---
## Dataset Preparation
Dataset link: https://huggingface.co/datasets/bookcorpus/bookcorpus

The project uses a custom `Library` class to generate and manage datasets:
- **Encoding**: Tokens are encoded using a defined `encoding` size.
- **Train/Test Split**: The dataset is divided into training and test sets.
- **DataLoader**: Provides sequential data in batches for training and evaluation.
- **Shannon**: Provides a Method to generate n tokens for a given model.
- While creating the library class, we specify the vocab size - based on the same it will choose a character level implementation or a byte pair implementation.


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

## Training and Evaluation
We compared the above three models in a series
of sizes.

For all models, Adam optimizer was used with a
negative log-likelihood loss function. Models were
trained up to 64 epochs. The set of nine models
were trained on the Discovery cluster at Northeastern, using 16GB of RAM, and using either a T4,
V100, or A100 GPU. 

For evaluation, we used the same starting conditions and the iteration count that minimized the perplexity.

---

## How to use the models
We have empirically tested different hyper parameters and set the best performing ones for all models in their respective notebook files. You can run all the models by just opening the notebook in the project root directory and running the cells. The hyper parameters, device management (GPU or CPU), and training, are all set in these notebooks. We have trained these models on T4,
V100, or A100 GPUs. Running the current parameters without access to proper GPU is not recommented. Lowering some dimensions, number of layers, should be enough to test the flow, learning, and generation of these models. The models available for use are `mlp.ipynb`, `s4.ipynb`, `transformer.ipynb`.

### Requirements
- Python 3.8+
- PyTorch 2.0+
- GPU with CUDA support (recommended)
