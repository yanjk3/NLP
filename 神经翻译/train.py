# -*- coding: utf-8 -*
"""
Author: Junkai-Yan
File: train.py
Finished in 2019/12/17
This file trains the model using dataset_10000(after pre-processing).
Model is in model.py.
Firstly, read in training data, training target and their dict.
Secondly, build the network, pay attention to the vocabulary size of encoder and decoder.
Thirdly, divided data set into some batches.
    For each batch, padding training data and label,
    such that its length equal to the longest sequence of this batch.
Fourthly, throw the batch training data into Encoder, and get the output and the last hidden state.
    In this part, because the LSTM is bidirectional, the output size of it is (seq_len, bs, 2*hidden_dim),
    It should be squeeze to (seq_len, bs, hidden_dim), the way I choose is to add the forward output and
    the backward output, so the output size becomes (seq_len, bs, hidden_dim).
Fifthly, init the initial hidden state of Decoder by the last hidden state of Encoder.
    In this part, the dimension problem discussed before takes place again, my operation is the same.
    Then for each time step, using random.random() to judge if we choose teacher forcing.
    If so, this time step we input the training target to Decoder, else, input the output of previous time step.
    For each time step, calculate the loss between output and target.
    For each batch, update the parameters.
    For each epoch, note down the average loss.
Lastly, save the total model as a 'pkl' file and show the variation of loss.
**********
What you should focus on is:
The dimension of input is not batch first,
so I transpose the input before throw it into Encoder.
What's more, the learning rate should not be too large,
otherwise, it will make the loss oscillate near the minimum.
**********
Hyper parameters:
    learning_rate = 1e-4
    decoder_learning_rate = 5e-4
    batch_size = 100
    epochs = 1000
    teaching_force_ratio = 0.5

Optimizer:
    Encoder: SGD
    Decoder: SGD

Loss function:
    Cross Entropy
"""
