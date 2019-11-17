import torch
from torch.autograd import Variable
from torch import nn, optim
import torch.nn.functional as F
import re
import jieba


word_to_idx = dict()
f = open('./分词结果和词表/dict.txt', 'r', encoding='utf-8')
for line in f:
    l = line.split()
    word_to_idx[l[1]] = int(l[0])
f.close()
word_to_idx['<BOS>'] = len(word_to_idx)
word_to_idx['<EOS>'] = len(word_to_idx)
idx_to_word = {word_to_idx[word]: word for word in word_to_idx}

stop_list = list()
f = open('停止词.txt', 'r')
for line in f:
    stop_list.append(line.strip())
f.close()

class Net(nn.Module):
    def __init__(self, vocb_size, context_size, n_dim):
        """
        :param vocb_size: number of non repeating words
        :param context_size: use how many words to predict a word
        :param n_dim: the dimension of embedded word
        """
        super(Net, self).__init__()
        self.n_word = vocb_size
        # nn.Embedding(m, n), m denotes number of words, n denotes dimension of a word
        self.layer1 = nn.Embedding(self.n_word, n_dim)
        self.layer2 = nn.Sequential(
            nn.Linear(context_size * n_dim, 128),
            nn.ReLU(True),
        )
        self.linear2 = nn.Linear(128, 64)
        self.linear3 = nn.Linear(64, self.n_word)

    def forward(self, x):
        batch_size, _ = x.size()
        emb = self.layer1(x)  # emb shape is (context_size, n_dim)
        emb = emb.view(batch_size, -1)  # change the shape to (BS, context_size * n_dim)
        out = self.layer2(emb)
        out = self.linear2(out)
        out = self.linear3(out)
        log_prob = F.log_softmax(out, dim=1)
        return log_prob

model = torch.load('./模型/3-gram-model.pkl').cuda()

f = open('questions.txt', 'r', encoding='utf-8')
fw = open('3-gram-answer.txt', 'w', encoding='utf-8')
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
        if len(input_list) == 2:
            break
    input_list.reverse()
    word = Variable(torch.LongTensor([input_list])).cuda()
    out = model(word)
    _, pred = torch.max(out, 1)  # 返回out中每一行的最大值并返回列号
    fw.write(idx_to_word[pred.item()]+'\n')
f.close()
fw.close()