import os
import json
import math
import operator
import time
import numpy as np

from random import shuffle

from model import LogisticReg
from options import Options
from process_data import get_data, DATA_SIZE, readFile, paths, readFile_test

if not os.path.isdir('models'):
    os.mkdir('models')

model = None
opt = Options()

def train():
    global model

    print("Training")
    
    since = time.time()

    data, vocab, label_map = readFile(paths[DATA_SIZE], 'train')
    print("Data size :{} Vocab size : {} labels size {}".format(len(data), len(vocab), len(label_map)))
    model = LogisticReg(len(vocab), len(label_map))

    for epoch in range(opt.EPOCH):
        # shuffle(data)
        running_acc = 0
        for labels, tokens in data:
            label_vec = label_to_vec(label_map, labels)
            token_vec = tokens_to_vec(vocab, tokens)
            # print(label_vec)
            # print(token_vec)
            running_acc = model.train_step(token_vec, label_vec)

        print("Epoch {}/{} accuracy : {:.4f}".format(epoch, opt.EPOCH, running_acc/len(data)))





    time_elapsed = time.time() - since
    print("Time for training {}".format(time_elapsed))

    with open('models/naive_bayes_data.json', 'w') as f:
        json.dump(naive_bayes_data, f)

    print()

    return naive_bayes_data

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

def get_model():
    with open('models/naive_bayes_data.json', 'r') as f:
        return json.load(f)

def devel(nb=None, m=1):
    
    if not nb:
        nb = get_model()

    print("Devel ")

    since = time.time()

    # data = readFile(paths[DATA_SIZE], 'devel')
    
    data = readFile(paths[DATA_SIZE], 'train')

    correct = 0;
    prob_y = {}
    # m = 1

    total_label_count = nb['total_label_count']
    label_count = nb['label_count']

    label_word_count = nb['label_word_count'] 
    label_any_word_count = nb['label_any_word_count']

    len_v = nb['len_vocab']
    dom_labels = nb['dom_labels']

    q_x = 1 / len_v
    q_y = 1 / len(dom_labels)
    
    # print(data[:4])
    for labels, words in data:
        for label in dom_labels:
            prob_y[label] = math.log( (label_count[label]+m*q_y) / (total_label_count+m) )
            for word in words:
                if word in label_word_count[label]:
                    num = label_word_count[label][word] + m*q_x
                else:
                    num = m*q_x
                den = label_any_word_count[label] + m
                prob_y[label] += math.log(num/den)
        max_label = max(prob_y, key=prob_y.get)

        if max_label in labels:
            correct +=1
    time_elapsed = time.time() - since


    print("Time for Validating {}".format(time_elapsed))
    print("Accuracy {:.4f}".format(correct/len(data)))
    
    print()
    return correct/len(data)


def test(nb=None, m=1):
    if not nb:
        nb = get_model()


    print("Testing  ")

    since = time.time()

    data, Ids = readFile_test(paths[DATA_SIZE], 'test')

    correct = 0;
    prob_y = {}
    # m = 1

    total_label_count = nb['total_label_count']
    label_count = nb['label_count']

    label_word_count = nb['label_word_count'] 
    label_any_word_count = nb['label_any_word_count']

    len_v = nb['len_vocab']
    dom_labels = nb['dom_labels']

    q_x = 1 / len_v
    q_y = 1 / len(dom_labels)


    for (labels, words), Id in zip(data, Ids):
        for label in dom_labels:
            prob_y[label] = math.log( (label_count[label]+m*q_y) / (total_label_count+m) )
            for word in words:
                if word in label_word_count[label]:
                    num = label_word_count[label][word] + m*q_x
                else:
                    num = m*q_x
                den = label_any_word_count[label] + m
                prob_y[label] += math.log(num/den)
        max_label = max(prob_y, key=prob_y.get)

        if max_label in labels:
            correct +=1

        # print("{}\t{}".format(Id, max_label))
        # print("{}\tground truth : {}".format(Id, labels))

    time_elapsed = time.time() - since

    print("Time for Testing {}".format(time_elapsed))


    print("Stat\t{} {}".format(correct, len(data)))
    print("Accuracy {:.4f}".format(correct/len(data)))
    print()




if __name__ == "__main__": 
    train()