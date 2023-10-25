import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences

maxlen = 256  # maximum sequence length
vocab_size = 20000  # vocabulary size

# Load the IMDb dataset
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=vocab_size)

# Pad sequences to a fixed length
x_train = pad_sequences(x_train, maxlen=maxlen, padding='post', truncating='post')
x_test = pad_sequences(x_test, maxlen=maxlen, padding='post', truncating='post')


# Create a Multi-Head Self-Attention Layer
class MultiHeadSelfAttention(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads):
        super(MultiHeadSelfAttention, self).__init__()
        self.num_heads = num_heads
        self.attention = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim // num_heads)
        self.norm = tf.keras.layers.LayerNormalization(epsilon=1e-6)

    def call(self, inputs):
        # Multi-Head Self-Attention
        attention_out = self.attention(inputs, inputs)
        # Residual Connection and Layer Normalization
        out = self.norm(inputs + attention_out)
        return out


embed_dim = 256  # embedding dimension
num_heads = 8  # number of attention heads

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(maxlen,)),
    tf.keras.layers.Embedding(vocab_size, embed_dim),
    MultiHeadSelfAttention(embed_dim, num_heads),
    # use Flatten if don't want to use Bidirectional LSTM
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
    #tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Model summary
model.summary()

# Compile the model
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# Train the model
model.fit(x_train, y_train, epochs=5, batch_size=32, validation_data=(x_test, y_test))

print("Model Evaluate:")
# Evaluate the model
model.evaluate(x_test, y_test)