import os
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.tokenize import RegexpTokenizer
import unicodedata
import re
import codecs
import pickle
import json

# Avaiable options : verysmall full
DATA_SIZE = 'verysmall'
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

            tokens = [token for token in sentence if token not in stopWords]
            # tokens = [token for token in word_tokenize(sentences[2]) if token not in stopWords]

            # vocab[dset_type].update(tokens)

            # Add label data pair to data.
            data += [(labels, tokens)]

            # Add labels to labels_map
            for label in labels:
                if label not in labels_map:
                    labels_map[label] = len(labels_map)

            add_to_vocab(vocab, tokens)

            count += 1
        return data, vocab, labels_map

def add_to_vocab(vocab, tokens):
    for token in tokens:
        if token not in vocab:
            vocab[token] = len(vocab)

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
    write_all_data()