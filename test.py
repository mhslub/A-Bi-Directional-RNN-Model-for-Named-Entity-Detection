import numpy as np
import tensorflow
from tensorflow.keras import Sequential, Model, Input, models
from tensorflow.keras.layers import LSTM, Embedding, concatenate, Dense, TimeDistributed, Bidirectional
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import CSVLogger
from tensorflow.keras.preprocessing.sequence import pad_sequences
from conlleval import evaluate

from preprocess import read_data, load_word_embeddings, generate_output_file

# =============================================================== #
#                       Read test data
# =============================================================== #
evaluation_dataset_type = 'test' #"dev" to read NER-de-dev.tsv, "test" to read NER-de-test.tsv

input_tokens, target_labels, _, _, _ = read_data(-1) #read training vocabulary
_, _, all_input_seq, all_target_seq1, all_target_seq2 = read_data(-1, dataset_type=evaluation_dataset_type) #read testing sequences
#add a padding token as part of the vocabulary
#padding index must be 0 to be masked by the Embedding layer
input_tokens.insert(0, 'PAD')
input_tokens.insert(1, 'UNKNOWN')
target_labels.insert(0, 'PAD')
target_labels.insert(1, 'UNKNOWN')#just in case a new tag appears in the dev dataset

num_input_tokens = len(input_tokens)
num_output_tokens = len(target_labels)
max_input_seq_length = 56#same as training
max_output_seq_length = 56
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
test_data = [[input_token_index.get(token.lower(), 1) for token in seq] for seq in all_input_seq]#must consider OOV cases
pre_first_level_output_indexes = [[target_token_index['O'] for label in seq] for seq in all_target_seq1]#to be used as inputs for first level predictions
first_level_output_indexes = [[target_token_index.get(label, 1) for label in seq] for seq in all_target_seq1]#first level Named Entities
second_level_output_indexes = [[target_token_index.get(label, 1) for label in seq] for seq in all_target_seq2]#second level (nested Named Entities)

#pad input and output sequences with PAD index
test_data = pad_sequences(maxlen=max_input_seq_length , sequences=test_data, padding='post', value=input_token_index['PAD'])
pre_first_level_output_indexes = pad_sequences(maxlen=max_output_seq_length , sequences=pre_first_level_output_indexes, padding='post', value=target_token_index["PAD"])
first_level_output_indexes = pad_sequences(maxlen=max_output_seq_length , sequences=first_level_output_indexes, padding='post', value=target_token_index["PAD"])
second_level_output_indexes = pad_sequences(maxlen=max_output_seq_length , sequences=second_level_output_indexes, padding='post', value=target_token_index["PAD"])

#convert label indexes into one-hot vectors (only for output)
first_level_output = [to_categorical(i, num_classes=num_output_tokens) for i in first_level_output_indexes]
second_level_output = [to_categorical(i, num_classes=num_output_tokens) for i in second_level_output_indexes]

# =============================================================== #
#                       Test The model
# =============================================================== #

#map indices back to their unique vocabulary words and target labels
reverse_input_token_index = dict((i, token) for token, i in input_token_index.items())
reverse_target_token_index = dict((i, label) for label, i in target_token_index.items())

#=========================First level prediction=================
model = models.load_model("model_bi_lstm/trained_model_level1.h5")

print('Running first level predictions...')
#test_data, target and predictions are of shape (sample_size, max_input_seq_length)
predictions = model.predict([test_data, pre_first_level_output_indexes])
predictions = np.argmax(predictions, axis=-1)
target = np.argmax(np.array(first_level_output), axis=-1)

print('Saving first level predictions to the output file...')

#exclude paddings
# mask_arr = [np.ones(len(seq)) for seq in all_input_seq]
# mask_arr_padded = pad_sequences(maxlen=max_input_seq_length , sequences=mask_arr, padding='post')

mask = (target > 0)#ignore padding
predicted_tag_ids = predictions[mask]
predicted_tags = [reverse_target_token_index[pred] for pred in predicted_tag_ids]

#generate the output file by appending the lines with corresponding predictions
#this file is used as input for the evaluation script

output_data_path = ""
if 'dev' in evaluation_dataset_type:
    dev_data_path = "data/NER-de-dev.tsv"
    output_data_path = "Evaluation/NER-de-dev_tagged.tsv"
elif 'test' in evaluation_dataset_type:
    dev_data_path = "data/NER-de-test.tsv"
    output_data_path = "Evaluation/NER-de-test_tagged.tsv"

generate_output_file(predicted_tags, dev_data_path, output_data_path)
# #=========================Second level prediction=================

model = models.load_model("model_bi_lstm/trained_model_level2.h5")

print('Running second level predictions...')
predictions = model.predict([test_data, first_level_output_indexes])
predictions = np.argmax(predictions, axis=-1)
# target = np.argmax(np.array(second_level_output), axis=-1)

print('Saving second level predictions to the output file...')
predicted_tag_ids = predictions[mask]
predicted_tags = [reverse_target_token_index[pred] for pred in predicted_tag_ids]

generate_output_file(predicted_tags, output_data_path, output_data_path)















