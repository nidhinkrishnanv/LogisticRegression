import os
import json
import math
import operator
import time
import numpy as np
import copy

from random import shuffle

from model import LogisticReg
from options import Options
from process_data import get_data, DATA_SIZE, readFile, paths, readFile_test, get_vocab, get_labels_map

if not os.path.isdir('models'):
    os.mkdir('models')

model = None
opt = Options()
vocab = get_vocab()
label_map = get_labels_map()

def train(train_data, devel_data, vocab, label_map, lr):
    """ Train the model on local data and validate on validation set. """
    global model

    print("Training")
    
    since = time.time()

    model = LogisticReg(len(vocab), len(label_map), lr)

    epoch_train_acc = []
    epoch_val_acc = []

    best_acc = 0
    best_epoch = 0
    best_model = None

    for epoch in range(opt.EPOCH):
        # shuffle(train_data)
        running_acc = 0

        # Set lr rate 
        # lr = decrease_lr(epoch) # decrease lr
        # lr = increase_lr(epoch) # increase lr

        # Train on train_data
        for token_vec, label_vec in get_data_batch(train_data, vocab, label_map):

            acc = model.train_step(token_vec, label_vec, lr)

            running_acc += acc

        train_acc = running_acc/len(train_data)#*opt.BATCH_SIZE
        epoch_train_acc.append(train_acc)

        # Validate on devel_data
        running_acc_val = 0
        for token_vec, label_vec in get_data_batch(devel_data, vocab, label_map):

            c_acc = model.test_step(token_vec, label_vec)

            running_acc_val += c_acc

        valid_acc = running_acc_val/len(devel_data)#*opt.BATCH_SIZE
        epoch_val_acc.append(valid_acc)

        if valid_acc > best_acc:
            best_acc = valid_acc
            best_epoch = epoch
            best_model = copy.deepcopy(model)

        print("Epoch {}/{} Train accuracy : {:.4f} Valid acc : {:.4f}".format(epoch, opt.EPOCH-1, train_acc, valid_acc))

    time_elapsed = time.time() - since
    print("Time for training {}".format(time_elapsed))
    print("Best accuray for LR : {} is {} for epoch {}".format(lr, best_acc, best_epoch))
    print("Training_acc {}".format(epoch_train_acc))
    print("Validation acc {}".format(epoch_val_acc))

    print()
    return best_model

def get_data_batch(data, vocab, label_map, tok_vec=np.zeros((len(vocab), opt.BATCH_SIZE)), lab_vec=np.zeros((len(label_map), opt.BATCH_SIZE))):
    shuffle(data)
    for s_idx in range(0, len(data) - opt.BATCH_SIZE + 1, opt.BATCH_SIZE):
        new_data = data[s_idx:(s_idx + opt.BATCH_SIZE)]
        tokens, labels = zip(*new_data)

        # (features, batch)
        tokens = tokens_to_vec_batch(vocab, tokens, tok_vec)

        # (label, batch)
        labels = label_to_vec_batch(label_map, labels, lab_vec)

        yield tokens, labels


def tokens_to_vec_batch(vocab, tokens, token_vec):
    # token_vec = np.zeros((len(vocab), opt.BATCH_SIZE))
    token_vec.fill(0)
    for idx, sentence in enumerate(tokens):
        for word in sentence:
            if word in vocab:
                token_vec[vocab[word]][idx] += 1
    # print(token_vec.shape)
    return token_vec

def label_to_vec_batch(label_map, labels, label_vec):
    # label_vec = np.zeros((len(label_map), opt.BATCH_SIZE))
    label_vec.fill(0)
    for idx, row in enumerate(labels):
        for label in row:
            label_vec[label_map[label]][idx] = 1
    # print(label_vec.shape)
    return label_vec


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

def test(test_data, vocab, label_map, model):
    """ Test the model """

    since = time.time()
    running_acc = 0
    for token_vec, label_vec in get_data_batch(test_data, vocab, label_map):

        c_acc = model.test_step(token_vec, label_vec)

        running_acc += c_acc

    test_acc = running_acc/len(test_data)

    time_elapsed = time.time() - since
    print("Time for testing {}".format(time_elapsed))


    print("Test acc : {:.4f}".format(test_acc))

def increase_lr(epoch, init_lr=1e-6):
    lr = init_lr * (10**(epoch//opt.lr_decay))
    if epoch%opt.lr_decay == 0:
        print("LR set to {}".format(lr))
    return lr

def decrease_lr(epoch, init_lr=1):
    lr = init_lr * (0.1**(epoch//opt.lr_decay))
    if epoch%opt.lr_decay == 0:
        print("LR set to {}".format(lr))    
    return lr


def tune_lr(train_data, devel_data, vocab, label_map):
    """ Used to choose best learning rate. """

    # Try with 10 to 10**-6
    lr_list = [10**x for x in range(-5, 2)]
    print("Running for learning rates {}".format(lr_list))
    for lr in lr_list:
        train(train_data, devel_data, vocab, label_map, lr)

if __name__ == "__main__":

    # Get the datasets
    print("Batch Size {} LR {} Data Type : {} Epochs : {} lr_decay : {}".format(opt.BATCH_SIZE, opt.lr, opt.DATA_SIZE, opt.EPOCH, opt.lr_decay))

    train_data, vocab, label_map = readFile(paths[DATA_SIZE], 'train')
    devel_data, _, _ = readFile(paths[DATA_SIZE], 'devel')
    test_data, _, _ = readFile(paths[DATA_SIZE], 'test')

    print("Train Data size :{} Valid data size {} Vocab size : {} labels size {}".format(len(train_data), len(devel_data), len(vocab), len(label_map)))
    

    # tune_lr(train_data, devel_data, vocab, label_map)
    
    model = train(train_data, devel_data, vocab, label_map, opt.lr)

    test(test_data, vocab, label_map, model)