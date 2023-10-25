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


# Create positional encoding
def positional_encoding(max_seq_len, d_model):
    angle_rads = get_angles(np.arange(max_seq_len)[:, np.newaxis], np.arange(d_model)[np.newaxis, :], d_model)
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    pos_encoding = angle_rads[np.newaxis, ...]
    return tf.cast(pos_encoding, dtype=tf.float32)

# Get positional angles
def get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
    return pos * angle_rates

# Scaled dot-product attention
def scaled_dot_product_attention(q, k, v, mask):
    matmul_qk = tf.matmul(q, k, transpose_b=True)
    d_k = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(d_k)
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
    output = tf.matmul(attention_weights, v)
    return output, attention_weights

# Multi-head Attention
def multi_head_attention(v, k, q, mask, d_model, num_heads):
    batch_size = tf.shape(q)[0]
    q = tf.keras.layers.Dense(d_model)(q)
    k = tf.keras.layers.Dense(d_model)(k)
    v = tf.keras.layers.Dense(d_model)(v)
    q = tf.reshape(q, (batch_size, -1, num_heads, d_model // num_heads))
    k = tf.reshape(k, (batch_size, -1, num_heads, d_model // num_heads))
    v = tf.reshape(v, (batch_size, -1, num_heads, d_model // num_heads))
    q = tf.transpose(q, perm=[0, 2, 1, 3])
    k = tf.transpose(k, perm=[0, 2, 1, 3])
    v = tf.transpose(v, perm=[0, 2, 1, 3])
    output, _ = scaled_dot_product_attention(q, k, v, mask)
    output = tf.transpose(output, perm=[0, 2, 1, 3])
    output = tf.reshape(output, (batch_size, -1, d_model))
    return output

# Position-wise feed-forward network
def feed_forward_network(x, d_model, dff):
    x = tf.keras.layers.Dense(dff, activation='relu')(x)
    x = tf.keras.layers.Dense(d_model)(x)
    return x

# Encoder layer
def encoder_layer(x, d_model, num_heads, dff, mask):
    attention_output = multi_head_attention(x, x, x, mask, d_model, num_heads)
    attention_output = tf.keras.layers.Dropout(0.1)(attention_output)
    output1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x + attention_output)
    ffn_output = feed_forward_network(output1, d_model, dff)
    ffn_output = tf.keras.layers.Dropout(0.1)(ffn_output)
    output2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)(output1 + ffn_output)
    return output2

# Encoder
def encoder(x, num_layers, d_model, num_heads, dff, mask):
    for _ in range(num_layers):
        x = encoder_layer(x, d_model, num_heads, dff, mask)
    return x

# Masking for padding tokens
def create_padding_mask(seq):
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)
    return seq[:, tf.newaxis, tf.newaxis, :]

# Transformer model
def transformer_model(input_shape, num_layers, d_model, num_heads, dff, vocab_size, maxlen):
    inputs = tf.keras.layers.Input(shape=input_shape)
    mask = tf.keras.layers.Lambda(lambda x: create_padding_mask(x))(inputs)
    x = tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=d_model)(inputs)
    x *= tf.math.sqrt(tf.cast(d_model, tf.float32))
    x += positional_encoding(maxlen, d_model)
    x = tf.keras.layers.Dropout(0.1)(x)
    x = encoder(x, num_layers, d_model, num_heads, dff, mask)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)
    model = tf.keras.models.Model(inputs, outputs)
    return model

# Initialize hyperparameters
num_layers = 4
d_model = 128
num_heads = 8
dff = 512

# Build the model
input_shape = (256,)
model = transformer_model(input_shape, num_layers, d_model, num_heads, dff, vocab_size, maxlen)

# Model summary
model.summary()

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, epochs=5, batch_size=32, validation_data=(x_test, y_test))

# Evaluate the model on the test set
print("Model Evaluate:")
model.evaluate(x_test, y_test)

# Save model
model.save("Transformer-imdb.h5")