---
title: Text Classification in Tensorflow
date: 2024-03-18
summary: Movie Review Sentiment classification using Tensorflow
---

In this tutorial, we demonstrate a text classification model from raw text from scratch based on the IMDB sentiment classification dataset.

```python
import os
os.environ["KERAS_BACKEND"] = "tensorflow"

import keras
import tensorflow as tf
import numpy as np
from keras import layers
```

Load Dataset
The dataset folder contains a train and test subfolder

```python
!curl -O https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz
!tar -xf aclImdb_v1.tar.gz

!dir aclImdb

## Chec content of the test directory
!dir aclImdb\test

## Chec content of the train directory
!dir aclImdb\train

# The train and test folders contains pos and negative subfolders for the positive and negative reviews. Lets see an example positive review from the train folder.

!type aclImdb\train\pos\6248_7.txt
```

Now we can use keras.utils.text_dataset_from_directory to generate a labeled tf.data.dataset object from a set of text files on disk filed into class-specific folders. We will use this to generate the training, validation dataset from the train directory using 80:20 split. We will use the test directory to generate the test dataset. 

```python
import keras

batch_size = 32
raw_train_ds = keras.utils.text_dataset_from_directory(
    "aclImdb/train",
    batch_size=batch_size,
    validation_split=0.2,
    subset="training",
    seed=1000
)

raw_val_ds = keras.utils.text_dataset_from_directory(
    "aclImdb/train",
    batch_size=batch_size,
    validation_split=0.2,
    subset="validation",
    seed=1000,
)

raw_test_ds = keras.utils.text_dataset_from_directory(
    "aclImdb/test",
    batch_size=batch_size
)

print(f"Number of batches in train dataset: {raw_train_ds.cardinality()}")
print(f"Number of batches in validation dataset: {raw_val_ds.cardinality()}")
print(f"Number of batches in test dataset: {raw_test_ds.cardinality()}")
```
Let's preview few samples of the data to ensure normalization and tokenization will work as expected. We use eager execution by evaluating the tensors using numpy().

```python
for text_batch, label_batch in raw_train_ds.take(1):
    for i in range(2):
        print(f"Text: {text_batch.numpy()[i]}")
        print(f"Label: {label_batch.numpy()[i]}")
```

Lets Prepare the data

```python
import string
import re
import tensorflow as tf
from tensorflow.keras import layers

# We create a custom standardization to handle the HTML break tags '<br />' since this cannot be removed by derault standardizer.
def custom_standardization(input_data):
    """
    - change to lower case
    - strip all html tags
    - remove punctuations
    """
    lower_case = tf.strings.lower(input_data)
    stripped_html = tf.strings.regex_replace(lower_case, "<br />", " ")
    return tf.strings.regex_replace(
        stripped_html, f"[{re.escape(string.punctuation)}]", ""
    )

# Define model constants
max_features = 20000 
sequence_length = 500 # we set an explicit maximum sequence length, since the CNNs model won't support variable sequence

# Create text vectorization layer to normalize, split, and map strings to integers.
# For this, we set out 'ouput_mode' to int'
vectorizer_layer = keras.layers.TextVectorization(
    standardize=custom_standardization,
    max_tokens=max_features,
    output_mode="int",
    output_sequence_length=sequence_length,
)

# First we extract the text-only dataset and call 'adapt' to create the vocabulary.
text_ds = raw_train_ds.map(lambda x, y: x)
# Call 'adapt' on text-only dataset
vectorizer_layer.adapt(text_ds)

# %%
from keras import layers

# Make the vectorization layer a part of the model
text_input = keras.Input(shape=(1,), dtype=tf.string, name='text')
x = vectorizer_layer(text_input)
x = layers.Embedding(max_features+1, embedding_dim)(x)

# %%
# Apply the vectorization layer to the raw dataset
def vectorize_text(text, label):
    text = tf.expand_dims(text, -1)
    return vectorizer_layer(text), label

# retrieve a batch of 32 reviews and labels as dataset
text_batch, label_batch = next(iter(raw_train_ds))
first_review, first_label = text_batch[0], label_batch[0]
print("Review", first_review)
print("Label", raw_train_ds.class_names[first_label])
print("Vectorized review", vectorize_text(first_review, first_label))

# %% [markdown]
# We can look up the token (string) that each integer corresponds to by calling .get_vocabulary() on the layer

# %%
print("1627 ---> ", vectorizer_layer.get_vocabulary()[1627])
print("313 ---> ", vectorizer_layer.get_vocabulary()[313])
print("Vocabulary ---> {}".format(len(vectorizer_layer.get_vocabulary())))

# %%
# Apply to dataset
train_ds = raw_train_ds.map(vectorize_text)
val_ds = raw_val_ds.map(vectorize_text)
test_ds = raw_test_ds.map(vectorize_text)

# Do async prefetching / buffering of the data for best GPU performance
"""
.cache() keeps data in memory after it's loaded off disk. 
This will ensure the dataset does not become a bottleneck while training your model. 
If your dataset is too large to fit into memory, you can also use this method to create a performant on-disk cache, 
which is more efficient to read than many small files.

.prefetch() overlaps data preprocessing and model execution while training.
"""
train_ds = train_ds.cache().prefetch(buffer_size=10)
val_ds = val_ds.cache().prefetch(buffer_size=10)
test_ds = test_ds.cache().prefetch(buffer_size=10)

# %% [markdown]
# #### Build and Train model

# %%
from tensorflow.keras import losses, layers

embedding_dim = 16
model_2 = tf.keras.Sequential([
    layers.Embedding(max_features, embedding_dim), # Embedding layer takes the integer-encoded reviews and looks up an embedding vector for each word-index. These vectors are learned as the model trains. The vectors add a dimension to the output array. The resulting dimensions are: (batch, sequence, embedding)
    layers.Dropout(0.2),
    layers.GlobalAveragePooling1D(), # GlobalAveragePooling1D layer returns a fixed-length output vector for each example by averaging over the sequence dimension. This allows the model to handle input of variable length, in the simplest way possible.
    layers.Dropout(0.2),
    layers.Dense(1, activation='sigmoid')
])

model_2.summary()

model_2.compile(loss=losses.BinaryCrossentropy(),
                optimizer='adam',
                metrics=[tf.metrics.BinaryAccuracy(threshold=0.5)])

epochs = 10
history = model_2.fit(train_ds,
                      validation_data=val_ds,
                      epochs=epochs)

# Evaluate the model
loss, accuracy = model_2.evaluate(test_ds)
print("Loss: ", loss)
print("Accuracy: ", accuracy)

# %%
# To visualize the training history
history_dict = history.history
acc = history_dict["binary_accuracy"]
val_acc = history_dict["val_binary_accuracy"]
loss = history_dict["loss"]
val_loss = history_dict["val_loss"]

epochs = range(1, len(acc)+1)

import matplotlib.pyplot as plt

# "bo" is for "blue dot"
plt.plot(epochs, loss, 'bo', label='Training loss')
# b is for "solid blue line"
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()

# %%
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')

plt.show()

# %% [markdown]
# #### Make an end-to-end model to export for inference on raw strings

# %%
export_model = tf.keras.Sequential([
    vectorizer_layer,
    model_2,
    layers.Activation("sigmoid")
])

export_model.compile(loss=losses.BinaryCrossentropy(from_logits=False),
                     optimizer="adam",
                     metrics=["accuracy"])

loss, accuracy = export_model.evaluate(raw_test_ds)
print(accuracy)

# %% [markdown]
# #### Inference on new examples

# %%
examples = tf.constant([
    "The movie was great",
    "The movie was okay.",
    "The movie was awful."
])

export_model.predict(examples)

```
