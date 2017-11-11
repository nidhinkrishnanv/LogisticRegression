import os
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.tokenize import RegexpTokenizer
import unicodedata
import re
import codecs
import pickle
import json
import numpy as np

from options import Options

opt = Options()


# Avaiable options : verysmall full
DATA_SIZE = opt.DATA_SIZE

# paths = {"verysmall":"/home/nidhinkrishnanv/workspace/Naive-Bayes-classifier/hadoop/DBPedia.verysmall/",
paths = {"verysmall":"/scratch/ds222-2017/assignment-1/DBPedia.verysmall/",
        "full":"/scratch/ds222-2017/assignment-1/DBPedia.full/"}

dset_types = ['train', 'test', 'devel']

stopWords = set(stopwords.words('english'))
myStopWords = set(['(', ')','', ' ', '``', '@', '.', 'en', ',', '–', '[', ']'])
stopWords.update(myStopWords)
tokenizer = RegexpTokenizer(r'\w+')


if not os.path.isdir('data'):
    os.mkdir('data')

if not os.path.isdir('data/'+DATA_SIZE):
    os.mkdir('data/'+DATA_SIZE)

# vocab = {dset_type:set() for dset_type in dset_types}

def readFile(path, dset_type):
    # print("processing "+ dset_type + " data...")
    labels_map = get_labels_map()
    vocab = get_vocab()
    word_count = get_word_count('train')
    with codecs.open(path+DATA_SIZE+'_' + dset_type + '.txt', encoding='unicode_escape', errors='ignore') as f:
        data = []
        data_list = []
        label_list = []
        count=0
        for line in f:

            # Skip the first three lines
            if count <3 and DATA_SIZE == 'verysmall':
                count += 1
                continue

            if count >= 1000 and opt.DEBUG:
                break

            sents = line.split("\t")

            labels = sents[0].split(",")
            labels[-1] = labels[-1].strip()
            sentences = sents[1].split(" ", 2)

            # Split data based on space and remove stop words
            tokens = []
            sentence = sentences[2].split()
            # sentence[0] = sentence[0].strip("\"")
            # sentence[-2] = sentence[-2].strip("\"@en")
            sentence = re.split("\W+", sentences[2])

            tokens = [token.lower() for token in sentence if token not in stopWords and token in vocab]
            # tokens = [token for token in word_tokenize(sentences[2]) if token not in stopWords]

            # vocab[dset_type].update(tokens)

            # Add label data pair to data.
            data.append((tokens, labels))

            # data_list.append(tokens_to_vec(vocab, tokens))
            # label_list.append(label_to_vec(labels_map, labels))


            # if dset_type == 'train':
            #     # Add labels to labels_map
            #     for label in labels:
            #         if label not in labels_map:
            #             labels_map[label] = len(labels_map)
                        
            #     add_to_vocab(vocab, tokens, word_count)

            count += 1
        return data, vocab, labels_map


def tokens_to_vec(vocab, tokens):
    token_vec = np.zeros(len(vocab))
    for word in tokens:
        if word in vocab:
            token_vec[vocab[word]] += 1
    return token_vec
    
def label_to_vec(label_map, labels):
    label_vec = np.zeros(len(label_map))
    for label in labels:
        label_vec[label_map[label]] = 1
    return label_vec

def add_to_vocab(vocab, tokens, word_count):
    for token in tokens:
        if token not in vocab:
            vocab[token] = len(vocab)


def create_vocab(path, dset_type):
    # print("processing "+ dset_type + " data...")
    word_count = {}
    labels_map = {}
    vocab = {}
    with codecs.open(path+DATA_SIZE+'_' + dset_type + '.txt', encoding='unicode_escape', errors='ignore') as f:
        data = []
        count=0
        for line in f:

            # Skip the first three lines
            if count <3 and DATA_SIZE == 'verysmall':
                count += 1
                continue

            sents = line.split("\t")

            labels = sents[0].split(",")
            labels[-1] = labels[-1].strip()
            sentences = sents[1].split(" ", 2)

            # Split data based on space and remove stop words
            tokens = []
            sentence = sentences[2].split()
            # sentence[0] = sentence[0].strip("\"")
            # sentence[-2] = sentence[-2].strip("\"@en")
            sentence = re.split("\W+", sentences[2])

            tokens = [token.lower() for token in sentence if token not in stopWords]
            # tokens = [token for token in word_tokenize(sentences[2]) if token not in stopWords]

            count += 1

            for label in labels:
                if label not in labels_map:
                    labels_map[label] = len(labels_map)
                        
            add_to_word_count(word_count, tokens)

        for word in word_count:
            if word_count[word] > opt.min_word_count:
                vocab[word] = len(vocab)

        print(len(vocab))

        with open('data/' + DATA_SIZE + '/' + dset_type + '_word_count.json', "w") as f:
            json.dump(word_count, f)

        with open('data/' + DATA_SIZE + '/' + dset_type + '_vocab.json', "w") as f:
            json.dump(vocab, f)

        with open('data/' + DATA_SIZE + '/' + dset_type + '_labels_map.json', "w") as f:
            json.dump(labels_map, f)

def get_word_count(dset_type):
    with open('data/' + DATA_SIZE + '/' + dset_type + '_word_count.json', "r") as f:
        return json.load(f)


def get_vocab(dset_type='train'):
    with open('data/' + DATA_SIZE + '/' + dset_type + '_vocab.json', "r") as f:
        return json.load(f)


def get_labels_map(dset_type='train'):
    with open('data/' + DATA_SIZE + '/' + dset_type + '_labels_map.json', "r") as f:
        return json.load(f)


def add_to_word_count(word_count, tokens):
    for token in tokens:
        if token not in word_count:
            word_count[token] = 0
        word_count[token] += 1

def write_all_data():
    for dset_type in dset_types:
        data, vocab, labels  = readFile(paths[DATA_SIZE], dset_type)
        # for label in labels:
        #     print("{} : {}".format(label, labels[label]))
        # print(data[:4])

        print("Processed {} dataset with data size {} and vocab size {}".format(dset_type, len(data), len(vocab[dset_type])))
        # Save the data
        with open('data/' + DATA_SIZE + '/' + dset_type + '.json', "w") as f:
            json.dump(data, f)
        with open('data/' + DATA_SIZE + '/' + dset_type + '.pkl', "wb") as f:
            pickle.dump(data, f)
    # print("words only in test {}".format(len(vocab['test'].difference(vocab['train']).difference(vocab['devel']))))

def get_train_data():
    with open('data/' + DATA_SIZE + '/' + 'train' + '.json', 'r') as  f:
        return pickle.load(f)

def get_data(d_size, dset_type):
    with open('data/' + d_size + '/' + dset_type + '.pkl', 'rb') as  f:
        return pickle.load(f)


def readFile_test(path, dset_type):
    # print("processing "+ dset_type + " data...")
    with codecs.open(path+DATA_SIZE+'_' + dset_type + '.txt', encoding='unicode_escape', errors='ignore') as f:
        data = []
        count=0
        Ids = []
        for line in f:

            # Skip the first three lines
            if count <3 and DATA_SIZE == 'verysmall':
                count += 1
                continue

            sents = line.split("\t")

            labels = sents[0].split(",")
            labels[-1] = labels[-1].strip()
            sentences = sents[1].split(" ", 2)


            Ids.append(sentences[0])

            # Split data based on space and remove stop words
            tokens = []
            sentence = sentences[2].split()
            sentence[0] = sentence[0].strip("\"")
            sentence[-2] = sentence[-2].strip("\"@en")
            # sentence = re.split("\W+", sentences[2])

            tokens = [token for token in sentence if token not in stopWords]
            # tokens = [token for token in word_tokenize(sentences[2]) if token not in stopWords]

            # Add label data pair to data.
            data += [(labels, tokens)]

            count += 1

        return data, Ids


if __name__ == "__main__":
    create_vocab(paths[DATA_SIZE], 'train')
    # train_data, vocab, label_map = readFile(paths[DATA_SIZE], 'train')
    # print(train_data[:4])