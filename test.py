
sentence = ['双十一', '盖楼', '领取', '二十亿', '红包']
word_to_number = {'<START>':0, '双十一':1, '盖楼':5, '领取':8, '二十亿':10, '红包':15, '<END>':30}
print('\033[0;31mSource sentence: '+str(sentence)+'\033[0m')

# create training set for n-gram (n=3)
print('*'*50)
print('\033[1;31mTraining set for n-gram (n=3)\033[0m')
print('Dictionary:', word_to_number)
sentence = ['<START>'] + sentence + ['<END>']
sequence = [word_to_number[word] for word in sentence]
print('Sentence after padding:', sentence)
print('Sentence after encoding:', sequence)
training_data = list()
training_label = list()
for i in range(len(sequence)-2):
    training_data.append([sequence[i], sequence[i+1]])
    training_label.append([sequence[i+2]])
print('Training data:', training_data)
print('Training label:', training_label)

# create training set for LSTM
print('*'*50)
print('\033[1;31mTraining set for LSTM\033[0m')
word_to_number[' '] = 7
print('Dictionary:', word_to_number)
print('Sentence after padding <START> and <END>:', sentence)

print('\033[0;31mIf limitation = 10 > len(sentence), padding space before sentence: \033[0m')
sentence1 = [' ']*(10-len(sentence)) + sentence
sequence1 = [word_to_number[word] for word in sentence1]
print('    New sentence: '+str(sentence1))
print('    New sequence: '+str(sequence1))
print('    Training data:', sequence1[:-1])
print('    Training label', sequence1[1:])

print('\033[0;31mIf limitation = 5 < len(sentence), cutting the tail of the sentence:\033[0m')
sentence1 = sentence[:5]
sequence1 = [word_to_number[word] for word in sentence1]
print('    New sentence: '+str(sentence1))
print('    New sequence: '+str(sequence1))
print('    Training data:', sequence1[:-1])
print('    Training label', sequence1[1:])

print('\033[0;31mIf limitation = 7 = len(sentence), do nothing:\033[0m')
print('    New sentence: '+str(sentence))
print('    New sequence: '+str(sequence))
print('    Training data:', sequence[:-1])
print('    Training label', sequence[1:])
