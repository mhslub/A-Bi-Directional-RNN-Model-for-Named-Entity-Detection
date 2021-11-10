import numpy as np
import tensorflow
from tensorflow.keras import Sequential, Model, Input, models
from tensorflow.keras.layers import LSTM, Embedding, concatenate, Dense, TimeDistributed, Bidirectional
from tensorflow.keras.utils import to_categorical, plot_model
from tensorflow.keras.callbacks import CSVLogger
from tensorflow.keras.preprocessing.sequence import pad_sequences
from conlleval import evaluate

from preprocess import read_data, load_word_embeddings, generate_output_file

# =============================================================== #
#                      Read training data
# =============================================================== #

input_tokens, target_labels, all_input_seq, all_target_seq1, all_target_seq2 = read_data(-1) #pass -1 to read all data samples
#add a padding token as part of the vocabulary
#padding index must be 0 to be masked by the Embedding layer
input_tokens.insert(0, 'PAD')
input_tokens.insert(1, 'UNKNOWN')
target_labels.insert(0, 'PAD')
target_labels.insert(1, 'UNKNOWN')

num_input_tokens = len(input_tokens)
num_output_tokens = len(target_labels)
max_input_seq_length = max([len(seq) for seq in all_input_seq])#56 is the longest sentence in training
max_output_seq_length = max([len(seq) for seq in all_target_seq1])
embedding_dim = 300 #embedding vector size


print ("Number of sentences loaded %s", len(all_input_seq))
print ("Number of unique tokens loaded %s", num_input_tokens)
print ("Number of unique labels loaded %s", num_output_tokens)
# print ("unique labels loaded %s", target_labels)
print("Max input sequence length", max_input_seq_length)
print("Max output sequence length:", max_output_seq_length)

#Build a dictionary to map unique vocabulary words and target labels to their indices
input_token_index = dict(zip(input_tokens, range(len(input_tokens))))
target_token_index = dict(zip(target_labels, range(len(target_labels))))


# =============================================================== #
#                Prepare input and output arrays
# =============================================================== #

#generate input and output 2D arrays
train_data = [[input_token_index[token.lower()] for token in seq] for seq in all_input_seq]
pre_first_level_output_indexes = [[target_token_index['O'] for label in seq] for seq in all_target_seq1]#to be used as inputs for first level predictions
first_level_output_indexes = [[target_token_index[label] for label in seq] for seq in all_target_seq1]#first level Named Entities
second_level_output_indexes = [[target_token_index[label] for label in seq] for seq in all_target_seq2]#second level (nested Named Entities)

#pad input and output sequences with PAD index
train_data = pad_sequences(maxlen=max_input_seq_length , sequences=train_data, padding='post', value=input_token_index['PAD'])
pre_first_level_output_indexes = pad_sequences(maxlen=max_output_seq_length , sequences=pre_first_level_output_indexes, padding='post', value=target_token_index["PAD"])
first_level_output_indexes = pad_sequences(maxlen=max_output_seq_length , sequences=first_level_output_indexes, padding='post', value=target_token_index["PAD"])
second_level_output_indexes = pad_sequences(maxlen=max_output_seq_length , sequences=second_level_output_indexes, padding='post', value=target_token_index["PAD"])

#convert label indexes into one-hot vectors (only for output)
first_level_output = [to_categorical(i, num_classes=num_output_tokens) for i in first_level_output_indexes]
second_level_output = [to_categorical(i, num_classes=num_output_tokens) for i in second_level_output_indexes]


# check input and output structure
# print(all_input_seq[0])
# print(train_data[0])
# print(all_target_seq1[0])
# print(len(first_level_output), len(first_level_output[0]))
# print(pre_first_level_output[0])


#load the embedding matrix for the input tokens
word_embedding_matrix = load_word_embeddings(num_input_tokens, input_token_index, embedding_dim)
#embedding of the labels (previous level outputs)
label_embedding_matrix = np.identity(num_output_tokens)

# =============================================================== #
#                         Define The model
# =============================================================== #

hidden_dim = 100 #hidden layer size for LSTM

word_inputs = Input(shape = (max_input_seq_length,), name='words_input')
word_embedding_layer = Embedding(input_dim=num_input_tokens, output_dim=embedding_dim,
                    input_length=max_input_seq_length, weights=[word_embedding_matrix], 
                    trainable=False, mask_zero=True)(word_inputs)


prev_level_labels_inputs = Input(shape = (max_input_seq_length,), name='prev_level_labels_inputs')
label_embedding_layer = Embedding(input_dim=num_output_tokens, output_dim=num_output_tokens,
                    input_length=max_input_seq_length, weights=[label_embedding_matrix], 
                    trainable=False, mask_zero=True)(prev_level_labels_inputs)

concatenated_input = concatenate([word_embedding_layer, label_embedding_layer])

model_bi_lstm = Bidirectional(LSTM(units=hidden_dim, return_sequences=True, dropout=0.2, recurrent_dropout=0.2))(concatenated_input)
# model_lstm = LSTM(units=hidden_dim, return_sequences=True, dropout=0.2, recurrent_dropout=0.2)(concatenated_input)
model_outputs = TimeDistributed(Dense(num_output_tokens,activation='softmax'))(model_bi_lstm)

#the complete model that converts input vectors (token index sequences) to output vectors (distribution probabilities)
model = Model([word_inputs, prev_level_labels_inputs], model_outputs)
model.compile(
    optimizer="adam", loss="categorical_crossentropy", metrics=['accuracy']
)
model.summary()
# plot_model(model, to_file='model_plot.png', show_shapes=True)

# =============================================================== #
#                       Train The model
# =============================================================== #

csv_logger = CSVLogger('training.log')

#train first level Named Entities
model.fit(
    [train_data, pre_first_level_output_indexes], np.array(first_level_output),
    batch_size=128,
    epochs=15,
    verbose = 1,
    validation_split=0.2, #20% of the samples are used to calculate the loss
    callbacks=[csv_logger]
)
# Save all models
model.save("model_bi_lstm/trained_model_level1.h5")

# models should be trained separately for better results
# it is better to comment the first level training before enabeling the second level training
# #train second level i.e. nested Named Entities
# model.fit(
#     [train_data, first_level_output_indexes], np.array(second_level_output),
#     batch_size=128,
#     epochs=7,
#     verbose = 1,
#     validation_split=0.2, #20% of the samples are used to calculate the loss
#     callbacks=[csv_logger]
# )

# # Save all models
# model.save("model_bi_lstm/trained_model_level2.h5")

# =============================================================== #
#            Test the model on sample from training
# =============================================================== #

#*****comment the training section and enable the following to test the trained model*****
# #map indices back to their unique vocabulary words and target labels
# reverse_input_token_index = dict((i, token) for token, i in input_token_index.items())
# reverse_target_token_index = dict((i, label) for label, i in target_token_index.items())

# model = models.load_model("model_bi_lstm/trained_model_level1.h5")

# sample_size = 10#len(train_data)
# #train_data, target and predictions are of shape (sample_size, max_input_seq_length)
# predictions = model.predict([train_data[0:sample_size], pre_first_level_output_indexes[0:sample_size]])
# predictions = np.argmax(predictions, axis=-1)
# target = np.argmax(np.array(first_level_output[0:sample_size]), axis=-1)


# print("{:20}\t {:5}\t {}\n".format("Token", "True", "Pred"))
# print("-" * 50)
# for i in range(len(train_data)):
#     for token ,label, pred in zip(train_data[i], target[i], predictions[i]):
#         if token != 0:#exclude padding
#             print("{:20}\t {:5}\t {}\n".format(reverse_input_token_index[token], reverse_target_token_index[label], reverse_target_token_index[pred]))
#     print("-" * 50)



