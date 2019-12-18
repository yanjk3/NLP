# -*- coding: utf-8 -*
"""
Author: Junkai-Yan
File: test.py
Finished in 2019/12/18
This file test the model and evaluate it by using BLEU-4.
Firstly, read in testing data, developing data and the dict of target language.
Secondly, load in the model have saved in training part.
Thirdly, for test and dev data, convert the sentences to sequences and throw them into model.
    In this part, I throw it one by one, surely you can throw a batch, but it need padding.(best not)
    We must not use teaching forcing in testing part.
    The input to the next time step is the output of previous time step, using beam search but not greedy.
    Choose the best sentence, that is, the sentence which has the highest probability to be the finally output,
    note it down in a file.
    Calculate the BLEU-4 score between target and output, also note the score down in the file.
**********
What you should focus on is:
The whole process is similar to training process, the differences are:
    1. when training, we use teaching forcing, but testing, no!
    2. when training, we use greedy to be the next time step's input, but testing, we use beam search
    3. when training, we calculate loss, but testing, we calculate BLEU-4 score.

This file your guys need to implement yourself.
    ^_^ have fun!
"""
