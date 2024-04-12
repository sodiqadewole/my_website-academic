---
title: Training Large Language Model from Scratch
summary: A tutorial of training LLM from scratch in a single notebook
tags:
  - Deep Learning, Large Language Model (LLM), Transformer Network, Multihead Attention
date: '2024-04-07T00:00:00Z'

# Optional external URL for project (replaces project detail page).
external_link: 'https://colab.research.google.com/drive/1G-atL1tgR6aM61AmuQr_1D4ZdlvcgAbZ?usp=sharing'

image:
  # caption: Photo by rawpixel on Unsplash
  # focal_point: Smart

# links:
#   - icon: twitter
#     icon_pack: fab
#     name: Follow
#     url: https://twitter.com/georgecushen
# url_code: ''
# url_pdf: ''
# url_slides: ''
# url_video: ''

# Slides (optional).
#   Associate this project with Markdown slides.
#   Simply enter your slide deck's filename without extension.
#   E.g. `slides = "example-slides"` references `content/slides/example-slides.md`.
#   Otherwise, set `slides = ""`.
# slides: example
---

```python
# -*- coding: utf-8 -*-
"""Building Large Language Model From Scratch in PyTorch

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1G-atL1tgR6aM61AmuQr_1D4ZdlvcgAbZ

### Part I: Data Preparation and Preprocessing

In this section we cover the data preparation and sampling to get our input data ready for the LLM. You can download our sample data from here: https://en.wikisource.org/wiki/The_Verdict
"""

with open("sample_data/the-verdict.txt", encoding="utf-8") as f:
    raw_text = f.read()

print(f"Total number of characters: {len(raw_text)}")
print(raw_text[:20]) # print the first 20 charaters

"""Next we tokenize and embed the input text for our LLM.
- First we develop a simple tokenizer based on some sample text that we then apply to the main input text above.
"""

import re
# Tokenize our input by splitting on whitespace and other characters
# Then we strip whitespace from each item and then filer out any empty strings
tokenized_raw_text = [item.strip() for item in re.split(r'([,.?_!"()\']|--|\s)', raw_text) if item.strip()]
print(len(tokenized_raw_text))
print(tokenized_raw_text[:20])

"""Next we convert the text tokens into token Ids that can be processed via embedding layers later. We can then build a vocabulary that consists of all the unique tokens."""

words = sorted(list(set(tokenized_raw_text)))
vocab_size = len(words)
print(f"Vocab size: {vocab_size}")

vocabulary = {token:integer for integer, token in enumerate(words)}

#Lets check the first 50 entries
for i, item in enumerate(vocabulary.items()):
    print(item)
    if i == 50:
        break

"""We can put these all together into our tokenizer class"""

class TokenizerLayer:
    def __init__(self, vocabulary):
        self.token_to_int = vocabulary
        self.int_to_token = {integer:token for token, integer in vocabulary.items()}

    # The encode function turns text into token ids
    def encode(self, text):
        encoded_text = re.split(r'([,.?_!"()\']|--|\s)', text)
        encoded_text = [item.strip() for item in encoded_text if item.strip()]
        return [self.token_to_int[token] for token in encoded_text]

    # The decode function turns token ids back into text
    def decode(self, ids):
        text = " ".join([self.int_to_token[i] for i in ids])
        # Replace spaces before the specified punctuations
        return re.sub(r'\s+([,.?!"()\'])', r'\1', text)

# Initialize and test tokenizer layer
tokenizer = TokenizerLayer(vocabulary)
print(tokenizer.encode(""""It's the last he painted, you know," Mrs. Gisburn said with pardonable pride."""))
print(tokenizer.decode(tokenizer.encode("""It's the last he painted, you know," Mrs. Gisburn said with pardonable pride.""")))

"""Next we special tokens for unknown words and to mark end of text.

SPecial tokens include:

[BOS] - Beginning of Sequence

[EOS] - End of Sequence. This markds the end of a text, usually used to concatenate multiple unrelated texts e.g. two different documents, wikipedia articles, books etc.

[PAD] - Padding: If we train an LLM with a batch size greater than 1, we may include multiple texts with different lenghts; with the padding token we pad the shorter texts to the longest length so that all texts have an equal lenght.

[UNK] - denotes words not included in the vocabulary
GPT2 only uses <|endoftext|> token for end of sequence and padding to reduce complexity which is analogous to [EOS].
Instead of <UNK> token for out-of-vocabulary words, GPT-2 uses byte-pair encoding (BPE) tokenizer, which breaks down words into subword unis.
For our application, we use <|endoftext|> tokens between two independent sources of text.
"""

tokenized_raw_text = [item.strip() for item in re.split(r'([,.?_!"()\']|--|\s)', raw_text) if item.strip()]
all_tokens = sorted(list(set(tokenized_raw_text)))
all_tokens.extend(["<|endoftext|>", "<|unk|>"])
vocabulary = {token:integer for integer, token in enumerate(all_tokens)}
tokenizer = TokenizerLayer(vocabulary)
print(len(tokenized_raw_text))
print(tokenized_raw_text[:20])

for i, item in enumerate(list(vocabulary.items())[-5:]):
    print(item)

# Get the new length of our vocabulary
print(len(vocabulary.items()))

class TokenizerLayer:
    def __init__(self, vocabulary):
        self.token_to_int = vocabulary
        self.int_to_token = {integer:token for token, integer in vocabulary.items()}

    # The encode function turns text into token ids
    def encode(self, text):
        encoded_text = re.split(r'([,.?_!"()\']|--|\s)', text)
        encoded_text = [item.strip() for item in encoded_text if item.strip()]
        encoded_text = [item if item in self.token_to_int else "<|unk|>" for item in encoded_text]
        return [self.token_to_int[token] for token in encoded_text]

    # The decode function turns token ids back into text
    def decode(self, ids):
        text = " ".join([self.int_to_token[i] for i in ids])
        # Replace spaces before the specified punctuations
        return re.sub(r'\s+([,.?!"()\'])', r'\1', text)

# Initialize and test tokenizer layer
tokenizer = TokenizerLayer(vocabulary)
print(tokenizer.encode(""""It's the last he painted, you know," Mrs. Gisburn said with pardonable pride."""))
print(tokenizer.decode(tokenizer.encode("""It's the last he painted, you know," Mrs. Gisburn said with pardonable pride.""")))

print(tokenizer.encode(""""This is a test! <|endoftext|> What is your favourite movie"""))
print(tokenizer.decode(tokenizer.encode("""This is a test! <|endoftext|> What is your favourite movie""")))

"""#### Byte Pair Encoding (BPE)
GPT-2 uses BPE as its tokenizer. This allows it to break down words that aren't in its predefined vocabulary into smaller subword units or even individual characters, enabling it to handle out-of-vocabulary words.

For example, if GPT-2's vocabulary doesn't have the word "unfamiliarword," it might tokenize it as ["unfam", "iliar", "word"] or some other subword breakdown, depending on its trained BPE merges

Original BPE Tokenizer can be found here: https://github.com/openai/gpt-2/blob/master/src/encoder.py


To use BPE tokenizer, we can use OpenAI's open-source tiktoken library which implements its core algorithms in Rust to improve computational performance.
"""

# pip install tiktoken

import tiktoken
import importlib

print("tiktoken version:", importlib.metadata.version("tiktoken"))

tokenizer = tiktoken.get_encoding("gpt2")
text = "Hello, this is a test sentence from theouterspace. <|endoftext|> It's the last he painted, you know,"
token_ids = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
print(token_ids)

# Re-construct the input text using the token_ids
print(tokenizer.decode(token_ids))

"""BPE tokenizer breaks down the unknown words into subwords and individual characters.

#### Data sampling with sliding window
We train LLM to generate one word at a time, so we want to prepare the training data accordingly where the next word in a sequence represents the target to predict:
"""

from IPython.display import Image
Image(url="https://drive.google.com/file/d/1-IpY_qgU0n704QJmoQYf8cAFIpeTuvTx/view?usp=sharing")

with open("sample_data/the-verdict.txt", "r") as f:
    raw_text = f.read()

encoded_text = tokenizer.encode(raw_text)
print(len(encoded_text))

"""- For each ext chunk, we want inputs and targets
- Since we want the model to predict the next word, the targets are the inputs shifted by one position to the right.
"""

sample = encoded_text[:100]
context_length = 5

for i in range(1, context_length + 1):
    context = sample[:i]
    desired_target = sample[i]
    print(context, "->", desired_target)

for i in range(1, context_length + 1):
    context = sample[:i]
    desired_target = sample[i]
    print(tokenizer.decode(context), "->", tokenizer.decode([desired_target]))

"""### Data Loading
Next we implement a simple data loader ha iterates over the input dataset and returns the inputs and target shifted by one.
"""

import torch
print("PyTorch version:", importlib.metadata.version("torch"))

"""- We use sliding window approach where we slide the window one word at a time (this is also called stride=1)
- We create a dataset and dataloader object that extract chunks from the input text dataset.
"""

from torch.utils.data import Dataset, DataLoader

class LLMDataset(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.tokenizer = tokenizer
        self.input_ids = []
        self.target_ids = []

        # Tokenize the entire text
        token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})

        # Iterate over the tokenized text
        for i in range(0, len(token_ids) - max_length, stride):
            context = token_ids[i:i+max_length]
            desired_target = token_ids[i+1:i+max_length+1]
            self.input_ids.append(torch.tensor(context))
            self.target_ids.append(torch.tensor(desired_target))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]


def create_data_loader(txt, batch_size=4, max_length=256, stride=128, shuffle=True, drop_last=True):
    # Initialize the tokenizer
    tokenizer = tiktoken.get_encoding("gpt2")

    # Create the dataset
    dataset = LLMDataset(txt, tokenizer, max_length, stride)

    # Create the data loader
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=drop_last)

with open("sample_data/the-verdict.txt", "r") as f:
    raw_text = f.read()

dataloader = create_data_loader(raw_text, batch_size=1, max_length=4, stride=1, shuffle=False)
data_iterator = iter(dataloader)
batch = next(data_iterator)
print(batch)

batch_2 = next(data_iterator)
print(batch_2)

# Increse the stride to remove overlaps between the batches since more overlap could lead to increased overfitting
dataloader = create_data_loader(raw_text, batch_size=8, max_length=4, stride=4, shuffle=False)

data_iter = iter(dataloader)
inputs, targets = next(data_iter)
print("Inputs:\n", inputs)
print("\nTargets:\n", targets)

"""#### Creating token embeddings
Next we embed the token in a continuous vector representation using an embedding layer. Usually the embedding layers are part of the LLM itself and are updated (trained) during model training.
"""

# Suppose we have the following four input examples with ids 5,1,3 and 2 after tokenization
input_ids = torch.tensor([[5, 1, 3, 2]])

"""For simplicity, suppose we have a small vocabulary of only 6 words and we want to create embeddings of size 3:"""

vocab_size = 6
embedding_size = 3

torch.manual_seed(42)
embedding_layer = torch.nn.Embedding(vocab_size, embedding_size)

# This would result in a 6x3 weight matrix
print(embedding_layer.weight)

"""The embedding output for our example input tensor will look as follows"""

embedding_layer(input_ids)

"""#### Encoding Word Positions

- Embedding layer convert Ids into identical vector representations regardless of where they are located in the input sequence.
- Positional embeddings are combined with the token embedding vector to form the input embedding for a large language model
- The BytePair encoder has a vocabulary size of 50,257
- To encode the input token to a 256-dimensional representation

"""

vocab_size = 50257
embedding_dim = 256

token_embedding_layer = torch.nn.Embedding(vocab_size, embedding_dim)

"""- if we sample data from the dataloader, we embed the tokens in each batch into a 256-dim vector
- if we have a batch size of 8 with 4 tokens each, this will result in a 8x4x256 tensor:
"""

max_length = 4
dataloader = create_data_loader(raw_text, batch_size=8, max_length=max_length, stride=max_length, shuffle=False)
data_iter = iter(dataloader)
inputs, targets = next(data_iter)
print("Token Ids:\n", inputs)
print("\nInputs shape:\n", inputs.shape)
print("\nEmbedding shape:\n", token_embedding_layer(inputs).shape)

"""- GPT-2 uses absolute position enbeddings, so we simply create another embedding layer

"""

context_length = max_length
pos_embedding_layer = torch.nn.Embedding(context_length, embedding_dim)

position_embeddings = pos_embedding_layer(torch.arange(context_length))
print(position_embeddings.shape)

"""- To create the input embeddings used in an LLM, we add the token and positional embeddings"""

input_embeddings = token_embedding_layer(inputs) + position_embeddings
print(input_embeddings.shape)

"""The illustration below shows the end-to-end preprocessing steps of input tokens to an LLM model."""
```