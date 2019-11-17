
word_to_idx = dict()
f = open('./分词结果和词表/dict.txt', 'r', encoding='utf-8')
for line in f:
    l = line.split()
    word_to_idx[l[1]] = int(l[0])
f.close()
word_to_idx['<BOS>'] = len(word_to_idx)
word_to_idx['<EOS>'] = len(word_to_idx)

gram_lib = dict()
for i in range(1000):
    f = open('./分词结果和词表/'+str(i + 1) + '.txt', 'r', encoding='utf-8')
    for line in f:
        sequence = [word_to_idx[word] for word in ['<BOS>']+line.split()+['<EOS>']]
        for j in range(len(sequence) - 2):
            temp = (sequence[j], sequence[j + 1], sequence[j + 2])
            if temp not in gram_lib:
                gram_lib[temp] = 1
            else:
                gram_lib[temp] += 1
    f.close()

f = open('gram_lib.txt', 'w', encoding='utf-8')
for e in gram_lib:
    f.write(str(e[0])+' '+str(e[1])+' '+str(e[2])+' '+str(gram_lib[e])+'\n')
f.close()