# -*- coding: utf-8 -*
"""
Author: Junkai-Yan
File: train.py
Finished in 2019/12/17
This file trains the model using dataset_10000(after pre-processing).
Model is in model.py.
Firstly, read in training data, training target and their dict.
Secondly, build the network, pay attention to the vacabulary size of encoder and decoder. 
Thirdly, divided data set into some batches.
    For each batch, padding training data such that its length equal to the
    longest sequence of this batch. What's more, padding training target to
    its length = max_length, which is a hyper parameter.
Thirdly, throw the training data into Encoder, and get the output and the last hidden state.
    In this part, the LSTM of the Encoder is bidirection, so the output should be reshaped
    reshaped to (seq_len, bs, hidden_dim), I add the
**********
What you should focus on is
"""

import torch
from torch import nn, optim
import random
import numpy as np
import matplotlib.pyplot as plt
from model import *

def get_data(filename, length=8000):
    data = list()
    f = open(filename, 'r', encoding='utf-8')
    for line in f:
        line = line.strip().split()
        data.append(line)
        if len(data) == length:
            break
    return data

def get_num2word(filename):
    data = list()
    f = open(filename, 'r', encoding='utf-8')
    for line in f:
        line = line.strip().split()
        data.append(line[1])
    return data

def padding_data(batch_data, num2word):
    new_batch_data = list()
    length = 0
    for data in batch_data:
        length = len(data) if len(data) > length else length
    for data in batch_data:
        if len(data) < length:
            data = data + [num2word.index('<PAD>')]*(length-len(data))
        new_batch_data.append(data)
    return np.array(new_batch_data, dtype='int64')

def padding_label(batch_data, num2word, length):
    new_batch_data = list()
    for data in batch_data:
        if len(data) < length:
            data = data + [num2word.index('<EOS>')]*(length-len(data))
        new_batch_data.append(data)
    return np.array(new_batch_data, dtype='int64')

if __name__=="__main__":
    training_data = get_data('./predata_10000/train_source_8000.txt', 300)
    training_label = get_data('./predata_10000/train_target_8000.txt', 300)
    source_num2word = get_num2word('./predata_10000/num2word_train_source_8000.txt')
    target_num2word = get_num2word('./predata_10000/num2word_train_target_8000.txt')

    # hyper parameter
    learning_rate = 1e-4
    decoder_learning_rate = 5e-4
    batch_size = 10
    epoches = 200
    max_length = 100
    teaching_force_ratio = 0.5

    # generate NN
    encoder = Encoder(len(source_num2word), 200, 500).cuda()
    decoder = Decoder(len(target_num2word), 200, 500).cuda()
    if encoder.hidden_dim != decoder.hidden_dim:
        raise RuntimeError('Encoder and Decoder should have the same hidden dimension!')

    # Optimizer is SGD
    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=decoder_learning_rate)

    # Loss function is cross entropy
    criterion = nn.CrossEntropyLoss()

    epoch_loss_list = list()
    for epoch in range(epoches):
        print('*' * 10)
        print('epoch: {} of {}'.format(epoch + 1, epoches))
        i = 0
        flag = False
        epoch_loss = 0.0
        # loop for batch
        while 1:
            loss = 0.0
            if (i + 1)*batch_size >= len(training_data):
                train_data = training_data[i*batch_size:]
                train_label = training_label[i*batch_size:]
                flag = True
            else:
                train_data = training_data[i*batch_size:(i + 1)*batch_size]
                train_label = training_label[i*batch_size:(i + 1)*batch_size]

            train_data = padding_data(train_data, source_num2word)
            train_label = padding_label(train_label, target_num2word, max_length)

            train_data = torch.LongTensor(train_data).transpose(0, 1).cuda()
            train_label = torch.LongTensor(train_label).transpose(0, 1).cuda()

            encoder_out, (h_n, h_c) = encoder(train_data)
            h_n = (h_n[0] + h_n[1]).unsqueeze(0)
            h_c = (h_c[0] + h_c[1]).unsqueeze(0)
            decoder_hidden = (h_n, h_c)
            # loop for each time step, feeding Decoder
            for time_step in range(max_length):
                # the first step, input '<BOS>'
                if time_step == 0:
                    begin_input = train_label[0].unsqueeze(0)
                    decoder_out, decoder_hidden = decoder(begin_input, decoder_hidden, encoder_out)
                # the rest time steps, using teacher forcing:
                else:
                    teacher_forcing = True if random.random() < teaching_force_ratio else False
                    if teacher_forcing:
                        time_step_input = train_label[time_step].unsqueeze(0)
                        decoder_out, decoder_hidden = decoder(time_step_input, decoder_hidden, encoder_out)
                    else:
                        _, time_step_input = torch.max(decoder_out, 2)
                        decoder_out, decoder_hidden = decoder(time_step_input, decoder_hidden, encoder_out)
                loss += criterion(decoder_out.squeeze(), train_label[time_step])
            encoder_optimizer.zero_grad()
            decoder_optimizer.zero_grad()
            loss.backward(retain_graph=True)
            encoder_optimizer.step()
            decoder_optimizer.step()
            epoch_loss += loss.item()/max_length
            if flag:
                break
            i += 1
        epoch_loss_list.append(epoch_loss)
        print(epoch_loss)

    torch.save(encoder, 'encoder.pkl')
    torch.save(decoder, 'decoder.pkl')

    plt.plot(epoch_loss_list, label='training loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    plt.show()
