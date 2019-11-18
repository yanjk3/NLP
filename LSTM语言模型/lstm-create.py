import torch
from torch.autograd import Variable
from torch import nn, optim
import torch.nn.functional as F
import random
import numpy as np
from numpy import *

word_to_idx = dict()
f = open('./分词结果和词表/dict.txt', 'r', encoding='utf-8')
for line in f:
    l = line.split()
    word_to_idx[l[1]] = int(l[0])
f.close()
word_to_idx[' '] = len(word_to_idx)
word_to_idx['<BOS>'] = len(word_to_idx)
word_to_idx['<EOS>'] = len(word_to_idx)
idx_to_word = {word_to_idx[word]: word for word in word_to_idx}

SEQ_LEN = 10
train = list()
for i in range(1000):
    f = open('./分词结果和词表/'+str(i + 1) + '.txt', 'r', encoding='utf-8')
    for line in f:
        if len(line.split()) == 0:
            continue
        sentence = ['<BOS>'] + line.split() + ['<EOS>']
        sequence = [word_to_idx[word] for word in sentence]
        if len(sequence) > SEQ_LEN:
            sequence = sequence[:SEQ_LEN]
        elif len(sequence) < SEQ_LEN:
            temp = [word_to_idx[' ']]*(SEQ_LEN-len(sequence))
            sequence = temp+sequence
        train.append(sequence)
    f.close()

BATCH_SIZE = 100
EPOCH = 300
EMBEDDING_DIM = 128

class Net(nn.Module):
    def __init__(self, vocb_size, n_dim, hidden_dim):
        """
        :param vocb_size: umber of non repeating words
        :param n_dim: the dimension of embedded word
        :param hidden_dim: the dimension of hidden layer
        """
        super(Net, self).__init__()
        self.n_word = vocb_size
        self.hidden_dim = hidden_dim
        self.layer1 = nn.Embedding(self.n_word, n_dim)
        self.layer2 = nn.LSTM(n_dim, self.hidden_dim, num_layers=2, batch_first=True)
        self.layer3 = nn.Sequential(
            nn.Linear(self.hidden_dim, 128),
            nn.ReLU(True),
        )
        self.layer4 = nn.Linear(128, vocb_size)

    def forward(self, x, hidden=None):
        batch_size, seq_len= x.size()
        if hidden is None:
            h_0 = x.data.new(2, batch_size, self.hidden_dim).fill_(0).float()
            c_0 = x.data.new(2, batch_size, self.hidden_dim).fill_(0).float()
        else:
            h_0, c_0 = hidden
        embeds = self.layer1(x) #(bs, seq_len) -> (bs, seq_len, n_dim)
        output, hidden = self.layer2(embeds, (h_0, c_0)) #(bs, seq_len, n_dim) -> (bs, seq_len, hidden_dim)
        output = output.reshape(seq_len*batch_size, self.hidden_dim) #(bs, seq_len, hidden_dim) -> (bs*seq_len, hidden_dim)
        output = self.layer3(output) #(bs*seq_len, hidden_dim) -> (bs*seq_len, 128)
        output = self.layer4(output) #(bs*seq_len, 128) -> (bs*seq_len, vocab_size)
        return output, hidden

model = Net(len(word_to_idx), EMBEDDING_DIM, 256).cuda()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(EPOCH):
    print('*' * 10)
    print('epoch: {} of {}'.format(epoch + 1, EPOCH))
    running_loss = 0.0
    running_acc = 0.0
    i = 0
    while (i + 1) * BATCH_SIZE < len(train):
        data = train[i * BATCH_SIZE:(i + 1) * BATCH_SIZE]
        data = np.array(data)
        word = Variable(torch.LongTensor(data[:, :-1])).cuda()
        label = Variable(torch.LongTensor(data[:, 1:])).cuda()
        label = label.view(-1)
        # forward
        out, _ = model(word)
        loss = criterion(out, label)
        running_loss += loss.data.item()
        _, pred = torch.max(out, 1)  # 返回out中每一行的最大值并返回列号
        num_correct = (pred == label).sum()
        running_acc += num_correct.item()
        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        i += 1
        if i % 200 == 0:
            print('[{}/{}] Loss: {:.6f}, Acc: {:.6f}'.format(
                epoch + 1, EPOCH, running_loss / (BATCH_SIZE * (i + 1)),
                running_acc / (BATCH_SIZE * (SEQ_LEN-1) *(i + 1))))
    random.shuffle(train)

torch.save(model, 'lstm-model.pkl')

