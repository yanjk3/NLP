import torch
from torch.autograd import Variable
from torch import nn
import re
import jieba

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

stop_list = list()
f = open('停止词.txt', 'r')
for line in f:
    stop_list.append(line.strip())
f.close()

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

model = torch.load('./模型/lstm-model.pkl').cuda()

f = open('questions.txt', 'r', encoding='utf-8')
fw = open('LSTM-answer.txt', 'w', encoding='utf-8')
for line in f:
    l = re.split('[、|。|？|！|；|：|，|（|）|~|——|“|”|…|\n]', line)
    sentence = ['<BOS>']
    for s in l:
        if len(s) == 0:
            continue
        s = jieba.cut(s)
        sentence += list(s)
    sentence += ['<EOS>']
    test = list()
    for s in sentence:
        if s not in stop_list:
            if s != '<BOS>' and s != '<EOS>':
                s = re.sub('\W', '', s)
            test.append(s)
    position = test.index('MASK')

    input_list = list()
    for i in range(position, -1, -1):
        if test[i] in word_to_idx:
            input_list.append(word_to_idx[test[i]])
    input_list.reverse()
    word = Variable(torch.LongTensor([input_list[:-1]])).cuda()
    out, _ = model(word)
    _, pred = torch.max(out, 1)  # 返回out中每一行的最大值并返回列号
    pred = pred.cpu().numpy()[-1]
    fw.write(idx_to_word[pred]+'\n')
f.close()
fw.close()