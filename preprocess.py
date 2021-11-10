from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from tensorflow.keras.utils import to_categorical

# =============================================================== #
#                    Process the training file
# =============================================================== #

#max_samples = -1 means read all samples
def read_data(max_samples, dataset_type='train'):
    all_input_seq = []
    all_target_seq1 = []
    all_target_seq2 = []
    input_tokens = set()
    target_labels = set()
 
    data_path = "data/NER-de-train.tsv"#training phase

    if 'dev' in dataset_type:
        data_path = "data/NER-de-dev.tsv"#evaluation phase (development)
        
    elif 'test' in dataset_type:
        data_path = "data/NER-de-test.tsv"#evaluation phase (test)

    with open(data_path, "r", encoding="utf-8") as f:
        lines = f.read().split("\n")

    input_seq = []
    target_seq1 = []
    target_seq2 = []
    max_samples = max_samples if max_samples > 0 else len(lines)
    for line in lines[0:max_samples]:
        line = line.strip()
        if len(line) > 0 and line[0] != '#':
            _, token, label1, label2 = line.split("\t")
            input_seq.append(token)

            target_seq1.append(label1)
            target_seq2.append(label2)

            #convert to lowercase (also must be considered to lookup the corresponding embeddings)
            if token.lower() not in input_tokens:
                input_tokens.add(token.lower())

            if label1 not in target_labels:
                target_labels.add(label1)
            
            if label2 not in target_labels:
                target_labels.add(label2)

        else:
            if len(input_seq) > 0:
                all_input_seq.append(input_seq)
                #add start and end padding to the target sequence
                all_target_seq1.append(target_seq1)
                all_target_seq2.append(target_seq2)

                #clear sequences
                input_seq = []
                target_seq1 = []
                target_seq2 = []

    #sort token so they always get same indexes 
    input_tokens = sorted(list(input_tokens))
    target_labels = sorted(list(target_labels))

    return input_tokens, target_labels, all_input_seq, all_target_seq1, all_target_seq2

# =============================================================== #
#    Generate Embeddings for the training data
#    Embedding are downloaded from: https://www.deepset.ai/german-word-embeddings (GloVe)
# =============================================================== #

def get_embedding(word, embeddings_index, embedding_dim):
    #Padding and Unknown words (out of vocabulary) will be given all zeros representation
    embedding_vector = [0] * embedding_dim
    if word == 'PAD':
        return embedding_vector
  
    if '-' not in word:
        return embeddings_index.get(word)#could be None (unknown words)
    else:
        #hyphenated tokens could be tagged with -part 
        #these elements are represented by the average embeddings of their tokens instead of UNKNOWN representation
        toks = word.split('-')
        sum_embedding_vector = embedding_vector
        for tok in toks:
            my_embed_vec = get_embedding(tok, embeddings_index, embedding_dim)
            if my_embed_vec is not None:
                sum_embedding_vector += my_embed_vec

        try:
            return sum_embedding_vector / float(len(toks))
        except:
            return embedding_vector


def load_word_embeddings(vocab_size, word_index, embedding_dim):
    path_to_vectors_file = 'vectors/vectors.txt'

    embeddings_index = {}

    print("loading word embeddings ...")
    with open(path_to_vectors_file) as f:
        for line in f:
            word, vec = line.split(maxsplit=1)
            vec = np.fromstring(vec, "f", sep=" ")
            embeddings_index[word] = vec


    print("Found total of %s word vectors." % len(embeddings_index))
    print("Prepare embedding matrix for the training data ...")
    embedding_matrix = np.zeros((vocab_size, embedding_dim))
    for word, i in word_index.items():
        embedding_vector = get_embedding(word, embeddings_index, embedding_dim)
        
        if embedding_vector is not None:
            #The embeddings of unknown words are all zeros (initial state of embedding_matrix)
            embedding_matrix[i] = embedding_vector



    return embedding_matrix


def load_label_embedding(vocab_size, label_index, num_output_tokens):
    label_embedding_matrix = np.zeros((vocab_size, num_output_tokens))
    for label, i in label_index.items():
        label_embedding_matrix[i] = to_categorical(i, num_classes=num_output_tokens)

    return label_embedding_matrix



#this is used to generate the output file for the evaluation script
def generate_output_file(predicted_tags, dev_data_path, output_data_path):
    with open(dev_data_path, "r", encoding="utf-8") as file_in:
        in_lines = file_in.read().split("\n")

    index = 0
    with open(output_data_path, "w", encoding="utf-8") as file_out:
        for line in in_lines:
            #line is tab separated
            line = line.strip()
            if len(line) > 0 and line[0] != '#': 
                if index < len(predicted_tags):
                    pred = predicted_tags[index]
                    file_out.write(line+"\t"+pred+"\n")
                    index+=1
                else:
                    file_out.write(line+"\n")
            else:
                file_out.write(line+"\n")


