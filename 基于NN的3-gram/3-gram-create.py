import torch
from torch.autograd import Variable
from torch import nn, optim
import torch.nn.functional as F
import random

word_to_idx = dict()
f = open('./分词结果和词表/dict.txt', 'r', encoding='utf-8')
for line in f:
    l = line.split()
    word_to_idx[l[1]] = int(l[0])
f.close()
word_to_idx['<BOS>'] = len(word_to_idx)
word_to_idx['<EOS>'] = len(word_to_idx)
idx_to_word = {word_to_idx[word]: word for word in word_to_idx}

trigram = list()
for i in range(1000):
    f = open('./分词结果和词表/'+str(i + 1) + '.txt', 'r', encoding='utf-8')
    for line in f:
        sentence = ['<BOS>'] + line.split() + ['<EOS>']
        sequence = [word_to_idx[word] for word in sentence]
        for j in range(len(sequence) - 2):
            trigram.append(((sequence[j], sequence[j + 1]), sequence[j + 2]))
    f.close()

BATCH_SIZE = 100
EPOCH = 300
CONTEXT_SIZE = 2
EMBEDDING_DIM = 200

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

model = Net(len(word_to_idx), CONTEXT_SIZE, EMBEDDING_DIM).cuda()
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(EPOCH):
    print('*' * 10)
    print('epoch: {} of {}'.format(epoch + 1, EPOCH))
    running_loss = 0.0
    running_acc = 0.0
    i = 0
    # for data in trigram:
    while (i + 1) * BATCH_SIZE < len(trigram):
        data = trigram[i * BATCH_SIZE:(i + 1) * BATCH_SIZE]
        word = list()
        label = list()
        for item in data:
            word.append([item[0][0], item[0][1]])
            label.append(item[1])
        word = Variable(torch.LongTensor(word)).cuda()
        label = Variable(torch.LongTensor(label)).cuda()
        # forward
        out = model(word)
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
        if i % 1000 == 0:
            print('[{}/{}] Loss: {:.6f}, Acc: {:.6f}'.format(
                epoch + 1, EPOCH, running_loss / (BATCH_SIZE * (i + 1)),
                running_acc / (BATCH_SIZE * (i + 1))))
    random.shuffle(trigram)
torch.save(model, '3-gram-model.pkl')

