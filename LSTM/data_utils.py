#!/home/athtrch/tensorflow_env/bin/python
"""
data_utils.py
-------------
This script contains data_utils used in the train script.
"""
import csv
import gensim
import nltk
import numpy as np
import string

def data_loader(path_to_file, path_to_embeddings, max_length):
    chf_cmplnt = list()
    ap = list()
    labels = list()
    labels_map = dict()
    count = 0
    with open(path_to_file) as csvfile:
        next(csvfile)
        for row in csvfile:
            temp = row.strip().split("\t")
            if temp != [""]:
                chf_cmplnt.append(nltk.word_tokenize(temp[0]))
                ap.append(nltk.word_tokenize(temp[1]))
                if temp[2][:3] not in labels_map:
                    labels_map[temp[2][:3]] = count
                    count += 1
                labels.append(labels_map[temp[2][:3]])
    embeddings, w2i = embeddings_loader(path_to_embeddings)
    chf_cmplnt = convert_to_index(chf_cmplnt, w2i, max_length[0])
    ap = convert_to_index(ap, w2i, max_length[1])
    matrix = np.zeros((len(labels), max(labels)+1))
    for i, l in enumerate(labels):
        matrix[i, l] = 1
    return chf_cmplnt, ap, matrix, embeddings, labels_map
    #return np.array(data), ids

def embeddings_loader(path_to_embeddings):
    """
    This function loads embeddings.
    """
    model = gensim.models.KeyedVectors.load_word2vec_format(path_to_embeddings, binary=True)
    w2v = np.zeros((len(model.wv.index2word), model.vector_size))
    w2i = dict()
    for i,word in enumerate(model.wv.index2word):
        w2v[i] = model[word]
        w2i[word] = i
    return w2v, w2i

def convert_to_index(data, w2i, max_length):
    """
    Converts Words to Indices interpretable by embeddings.
    """
    output = list()
    for sentence in data:
        temp = list()
        for word in sentence:
            if word in w2i:
                temp.append(w2i[word])
            #else:
            #    temp.append(w2i["UNK"])
        if len(temp) < max_length:
            temp += [0] * int(max_length - len(temp))
        output.append(temp[:max_length])
    return np.array(output)

def iterator(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            if (epoch != 0) and (batch_num == 0):
                dev = True
            else:
                dev = False
            yield shuffled_data[start_index:end_index], dev
