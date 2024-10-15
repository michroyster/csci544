import pandas as pd
import numpy as np
import torch
from torch.utils import data
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence, pad_packed_sequence
from sklearn import metrics
import gzip, sys

mode = -1
if len(sys.argv) != 2:
    print("\t>> Incorrect Usage. Must specify predict or train")
    print("\t\t> Correct usage for predicting is: python Royster_Michael_hw4.py predict")
    print("\t\t> Correct usage for training is  : python Royster_Michael_hw4.py train")
    exit()
if sys.argv[1] == 'predict':
    print("Beginning in PREDICT mode\n\n")
    mode = 0
elif sys.argv[1] == 'train':
    print("Beginning in TRAIN mode.\n\n")
    mode = 1
else:
    print("Unexpected...")
    exit()


def load_data(filename):
    num, word, tag = [], [], []
    with open(filename, 'r') as file:
        for i, line in enumerate(file):
            values = line.split()
            if values:
                num.append(values[0])
                word.append(values[1])
                tag.append(values[2])

    temp = {'position' : num, 'word' : word, 'ner_tag' : tag}
    data = pd.DataFrame(temp)
    return data

def load_batch(filename, pos):
    sentences = []
    with open(filename, 'r') as file:
        s = []
        for i, line in enumerate(file):
            values = line.split()
            if values:
                s.append(values[pos])
            else:
                sentences.append(s)
                s = []
        sentences.append(s)
        return sentences

def convert_to_idx(raw, lookup):
    full_idx = []
    for sentence in raw:
        temp = []
        for word in sentence:
            temp.append(lookup.get(word, len(lookup)-1)) # TODO ADD <unk> TO VOCAB AND MAKE IT RETURN <unk> INDEX INSTEAD OF 0
        full_idx.append(temp)
    return full_idx

def predict(model, device, loader, verbose=False):
    pred, actual = [], []
    # model.to(device)
    model.eval()
    with torch.no_grad():
        for x, y, l in loader:
            x = x.to(device)
            y = pack_padded_sequence(y, l, batch_first=True, enforce_sorted=False)
            labels, _ = pad_packed_sequence(y, batch_first=True)
            y = y.to(device)
            output = model(x, l)
            _, predictions = torch.max(output, dim=2)
            # print(predictions.size())
            for i in range(predictions.shape[0]):
                pred.append(predictions[i].tolist())
                actual.append(labels[i].tolist())

    return pred, actual

""" Utilities """
def back_to_tags(output, vocab):
    decoded = []
    for sentence in output:
        temp = []
        for num in sentence:
            temp.append(vocab[num])
        decoded.append(temp)
    return decoded

def write_prediction(output, inputs, filename):
    with open(filename, 'w') as file:
        for x, y in zip(output, inputs):
            for i in range(len(x)):
                file.write(f"{i} {y[i]} {x[i]}\n")
            file.write("\n")

def write_prediction_perl(output, inputs, labels, filename):
    with open(filename, 'w') as file:
        for x, y, z in zip(output, inputs, labels):
            for i in range(len(x)):
                file.write(f"{i} {y[i]} {z[i]} {x[i]}\n")
            file.write("\n")

def get_cap_bit(raw):
    cap_bit = []
    for sentence in raw:
        temp = []
        for word in sentence:
            if word[0].isupper():
                temp.append(1)
            else:
                temp.append(0)
        cap_bit.append(temp)
    return cap_bit

def glove_convert_to_idx(raw, lookup):
    full_idx = []
    for sentence in raw:
        temp = []
        for word in sentence:
            word = word.lower()
            temp.append(lookup.get(word, len(lookup)-1))
        full_idx.append(temp)
    return full_idx

def new_glove_convert_to_idx(raw, lookup):
    full_idx = []
    for sentence in raw:
        temp = []
        for word in sentence:
            # word = word.lower()
            temp.append(lookup.get(word, 0))
        full_idx.append(temp)
    return full_idx

def glove_predict(model, device, loader, verbose=False):
    pred, actual = [], []
    model.to(device)
    model.eval()
    with torch.no_grad():
        for x, y, l, c in loader:
            x = x.to(device)
            y = pack_padded_sequence(y, l, batch_first=True, enforce_sorted=False)
            labels, _ = pad_packed_sequence(y, batch_first=True)
            y = y.to(device)
            c = c.to(device)
            output = model(x, l, c)
            _, predictions = torch.max(output, dim=2)
            # print(predictions.size())
            for i in range(predictions.shape[0]):
                pred.append(predictions[i].tolist())
                actual.append(labels[i].tolist())

    return pred, actual

def new_glove_predict(model, device, loader, verbose=False):
    pred, actual = [], []
    model.to(device)
    model.eval()
    with torch.no_grad():
        for x, y, l, c, x_char in loader:
            x = x.to(device)
            x_char = x_char.to(device)
            y = pack_padded_sequence(y, l, batch_first=True, enforce_sorted=False)
            labels, _ = pad_packed_sequence(y, batch_first=True)
            y = y.to(device)
            c = c.to(device)
            output = model(x, x_char, l, c)
            _, predictions = torch.max(output, dim=2)
            # print(predictions.size())
            for i in range(predictions.shape[0]):
                pred.append(predictions[i].tolist())
                actual.append(labels[i].tolist())

    return pred, actual

class NER_Data(data.Dataset):
    def __init__(self, x, y):

        self.lengths = torch.tensor([len(s) for s in x])
        self.x = pad_sequence([torch.tensor(s) for s in x], batch_first=True)
        self.y = pad_sequence([torch.tensor(s) for s in y], batch_first=True, padding_value=-1)

    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, index):
        return self.x[index], self.y[index], self.lengths[index]

class Glove_Data(data.Dataset):
    def __init__(self, x, y, z):

        self.lengths = torch.tensor([len(s) for s in x])
        self.x = pad_sequence([torch.tensor(s) for s in x], batch_first=True)
        self.y = pad_sequence([torch.tensor(s) for s in y], batch_first=True, padding_value=-1)
        self.z = pad_sequence([torch.tensor(s) for s in z], batch_first=True, padding_value=-1)

    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, index):
        return self.x[index], self.y[index], self.lengths[index], self.z[index]

class BLSTM_CNN_Data(data.Dataset):
    def __init__(self, x, y, cap_bit, char_vocab, max_len, word_vocab):

        self.lengths = torch.tensor([len(s) for s in x])
        self.x = pad_sequence([torch.tensor(s) for s in x], batch_first=True)
        self.y = pad_sequence([torch.tensor(s) for s in y], batch_first=True, padding_value=-1)
        self.cap_bit = pad_sequence([torch.tensor(s) for s in cap_bit], batch_first=True, padding_value=-1)
        char_temp = []
        for s in x:
            temp=[]
            for w in s:
                temp.append(self.word_to_char_feature(word_vocab[w], max_len, char_vocab))
            char_temp.append(torch.tensor(temp))

        self.char_x = pad_sequence(char_temp, batch_first=True)

    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, index):
        return self.x[index], self.y[index], self.lengths[index], self.cap_bit[index], self.char_x[index]
    
    def word_to_char_feature(self,word, max_len, char_vocab):
        n = len(str(word))
        first_pad = int((max_len - n) / 2)
        second_pad = max_len - n - first_pad
        return [0 for i in range(first_pad-1)] + [1] + [char_vocab.get(c,-1) for c in word] + [2] + [0 for i in range(second_pad-1)]
        # return [0 for i in range(first_pad)] + [char_vocab.get(c,-1) for c in word] + [0 for i in range(second_pad)]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
raw_data = load_data('data/train')

threshhold = 3

vocabulary = raw_data.groupby(["word"]).size().reset_index(name="occurences")
vocabulary.sort_values(by='occurences', inplace=True, ascending=False)
vocabulary.reset_index(drop=True, inplace=True)

reduced = vocabulary.query("occurences > @threshhold")

tag_weights = raw_data['ner_tag'].value_counts().to_dict()
tag_total = 0
for key in tag_weights:
    tag_total += tag_weights[key]

words = reduced['word'].unique()
ner_tags = raw_data['ner_tag'].unique()

vocab = dict()
vocab_lookup = dict()
for i, word in enumerate(words):
    vocab[i] = word
    vocab_lookup[word] = i
vocab_len = len(vocab)
vocab[vocab_len] = "<unk>"
vocab_lookup["<unk>"] = vocab_len
vocab_len = len(vocab)

ner_tag_dict = dict()
ner_tags_lookup = dict()
for i, tag in enumerate(ner_tags):
    ner_tag_dict[i] = tag
    ner_tags_lookup[tag] = i
ner_tag_len = len(ner_tags)

train_input_raw = load_batch('data/train', 1)
dev_input_raw = load_batch('data/dev', 1)

test_input_raw = load_batch('data/test', 1)

train_label_raw = load_batch('data/train', 2)
dev_label_raw = load_batch('data/dev', 2)

train_input = convert_to_idx(train_input_raw, vocab_lookup)
dev_input = convert_to_idx(dev_input_raw, vocab_lookup)
test_input = convert_to_idx(test_input_raw, vocab_lookup)

train_lengths = [len(x) for x in train_input]
dev_lengths = [len(x) for x in dev_input]
test_lengths = [len(x) for x in test_input]

train_label = convert_to_idx(train_label_raw, ner_tags_lookup)
dev_label = convert_to_idx(dev_label_raw, ner_tags_lookup)

test_label = []
for s in test_input:
    test_label.append([0 for i in range(len(s))])


key = {0 : [1,0,0,0,0,0,0,0,0],
        1 : [0,1,0,0,0,0,0,0,0],
        2 : [0,0,1,0,0,0,0,0,0],
        3 : [0,0,0,1,0,0,0,0,0],
        4 : [0,0,0,0,1,0,0,0,0],
        5 : [0,0,0,0,0,1,0,0,0],
        6 : [0,0,0,0,0,0,1,0,0],
        7 : [0,0,0,0,0,0,0,1,0],
        8 : [0,0,0,0,0,0,0,0,1],
        9 : [0,0,0,0,0,0,0,0,0]
        }
new_train_label = []
for i in range(len(train_label)):
    temp = []
    for label in train_label[i]:
        temp.append(key[label])
    new_train_label.append(temp)

new_dev_label = []
for i in range(len(dev_label)):
    temp = []
    for label in dev_label[i]:
        temp.append(key[label])
    new_dev_label.append(temp)

for x, y in zip(new_dev_label, dev_input):
    if len(x) != len(y):
        print("ERROR",x ,y)
        break

class BLSTM(torch.nn.Module):
    def __init__(self, vocab_n, embedding_dim, hidden_dim, linear_dim, D_out, dropout):
        super(BLSTM, self).__init__()
        self.embed = torch.nn.Embedding(vocab_n, embedding_dim=embedding_dim)
        self.lstm = torch.nn.LSTM(embedding_dim, hidden_dim, bidirectional=True, batch_first=True)
        self.d = torch.nn.Dropout(p=dropout)
        self.linear = torch.nn.Linear(hidden_dim*2, linear_dim)
        self.elu = torch.nn.ELU()
        self.classifier = torch.nn.Linear(linear_dim, D_out, dtype=torch.float32)
        self.weight_init()

    def forward(self, text, lengths):
        output = self.embed(text)

        output = pack_padded_sequence(output, lengths, batch_first=True, enforce_sorted=False)
        output, (h, c) = self.lstm(output)
        output, x = pad_packed_sequence(output, batch_first=True)

        output = self.d(output)
        output = self.linear(output)
        output = self.elu(output)
        output = self.classifier(output)

        return output
    
    def weight_init(self):
        torch.nn.init.xavier_uniform_(self.linear.weight)
        torch.nn.init.xavier_uniform_(self.classifier.weight)

first_model = BLSTM(vocab_len, 100, 256, 128, ner_tag_len, 0.33)

weights = []
for key, value in ner_tag_dict.items():
    weights.append(1-(tag_weights[value]/tag_total))

weights = torch.tensor(weights)
weights = weights.to('cuda')

if mode == 1:
    loss_fn = torch.nn.CrossEntropyLoss(weight=weights.float(), ignore_index=-1)
    learning_rate = 0.1 # best=0.1
    batch_size = 4 # best=4

    optimizer = torch.optim.SGD(first_model.parameters(), lr=learning_rate, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

    x_train = NER_Data(train_input, train_label)
    train_loader = data.DataLoader(dataset=x_train, batch_size=batch_size, shuffle=True)
    first_model.cuda()

    best_running_loss = 10_000
    running_loss = 0

    epochs = 50
    for e in range(epochs):
        for i , (x, y, l) in enumerate(train_loader):
            # test = x.reshape(batch_size,20,300).to(device)
            x = x.to(device)
            y = pack_padded_sequence(y, l, batch_first=True, enforce_sorted=False)
            y, _ = pad_packed_sequence(y, batch_first=True)
            y = y.to(device)

            output = first_model(x, l)
            output = torch.permute(output, (0,2,1))

            loss = loss_fn(output, y)

            running_loss += loss.item() / x.size(0)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()
        
        if (running_loss < best_running_loss):
            best_running_loss = running_loss
            stagnate = 0
        else:
            stagnate += 1
        
        if stagnate > 2:
            break

        if e % 2 == 0:
            print(f"Epoch: {e} || Loss: {running_loss}")
        running_loss = 0
    torch.save(first_model, 'blstm1.pt')
elif mode == 0:
    print("="*10, "Loading blstm1.pt", "="*10)
    first_model = torch.load('blstm1.pt')
    # first_model = first_model.load_state_dict(torch.load('blstm1.pt'))

x_train = NER_Data(train_input, train_label)
train_loader = data.DataLoader(dataset=x_train, batch_size=1)
train_pred, train_actual = predict(first_model, device, train_loader, verbose=True)
test_output = back_to_tags(train_pred, ner_tag_dict)

write_prediction_perl(test_output, train_input_raw, train_label_raw, "train_prediction1.txt")

flat_x, flat_y = [], []
for output, label in zip(train_pred, train_actual):
    for i in range(len(output)):
        flat_x.append(output[i])
        flat_y.append(label[i])

#204567
print(metrics.classification_report(flat_y, flat_x))

x_dev = NER_Data(dev_input, dev_label)
dev_loader = data.DataLoader(dataset=x_dev, batch_size=1)
dev_pred, dev_actual = predict(first_model, device, dev_loader, verbose=True)
dev_output = back_to_tags(dev_pred, ner_tag_dict)

write_prediction(dev_output, dev_input_raw, "dev1.out")
write_prediction_perl(dev_output, dev_input_raw, dev_label_raw, "dev_prediction1.txt")

flat_x, flat_y = [], []
for output, label in zip(dev_pred, dev_actual):
    for i in range(len(output)):
        flat_x.append(output[i])
        flat_y.append(label[i])

#204567
print(metrics.classification_report(flat_y, flat_x))

x_test = NER_Data(test_input, test_label)
test_loader = data.DataLoader(dataset=x_test, batch_size=1)
test_pred, test_actual = predict(first_model, device, test_loader, verbose=True)
test_output = back_to_tags(test_pred, ner_tag_dict)

write_prediction(test_output, test_input_raw, "test1.out")

glove = "glove.6B.100d.gz"

glove_vocab = {}
glove_vocab_lookup = {}
glove_embeddings = []

with gzip.open(glove, "rt", encoding='utf-8') as g_file:
    for i, line in enumerate(g_file):
        temp = line.split()
        glove_vocab[temp[0]] = i
        glove_vocab_lookup[i] = temp[0]
        glove_embeddings.append(list(map(float, temp[1:])))

unk_index = len(glove_vocab)
glove_vocab['<unk>'] = unk_index
glove_vocab_lookup[unk_index] = '<unk>'
glove_embeddings.append([0 for i in range(100)])

glove_train_input = glove_convert_to_idx(train_input_raw, glove_vocab)
glove_dev_input = glove_convert_to_idx(dev_input_raw, glove_vocab)
glove_test_input = glove_convert_to_idx(test_input_raw, glove_vocab)

train_cap_bit = get_cap_bit(train_input_raw)
dev_cap_bit = get_cap_bit(dev_input_raw)
test_cap_bit = get_cap_bit(test_input_raw)

class Glove_BLSTM(torch.nn.Module):
    def __init__(self, word_embedding_features, embedding_dim, hidden_dim, linear_dim, D_out, dropout):
        super(Glove_BLSTM, self).__init__()
        self.embed = torch.nn.Embedding.from_pretrained(word_embedding_features)
        self.lstm = torch.nn.LSTM(embedding_dim+1, hidden_dim, bidirectional=True, batch_first=True)
        self.d = torch.nn.Dropout(p=dropout)
        self.linear = torch.nn.Linear(hidden_dim*2, linear_dim)
        self.elu = torch.nn.ELU()
        self.classifier = torch.nn.Linear(linear_dim, D_out, dtype=torch.float32)
        self.weight_init()

    def forward(self, text, lengths, cap):
        output = self.embed(text)

        cap = cap.unsqueeze(2)
        output = torch.cat([output, cap], dim=2)

        output = pack_padded_sequence(output, lengths, batch_first=True, enforce_sorted=False)
        output, (h, c) = self.lstm(output)
        output, x = pad_packed_sequence(output, batch_first=True)

        output = self.d(output)
        output = self.linear(output)
        output = self.elu(output)
        output = self.classifier(output)

        return output
    
    def weight_init(self):
        torch.nn.init.xavier_uniform_(self.linear.weight)
        torch.nn.init.xavier_uniform_(self.classifier.weight)




weights = []
for key, value in ner_tag_dict.items():
    weights.append(1-(tag_weights[value]/tag_total))
weights = torch.tensor(weights)
weights = weights.to('cuda')
second_model = Glove_BLSTM(torch.tensor(glove_embeddings), 100, 256, 128, ner_tag_len, 0.33)

if mode == 1:
    loss_fn = torch.nn.CrossEntropyLoss(weight=weights.float(), ignore_index=-1)
    learning_rate = 0.1 # best=0.1
    batch_size = 4 # best=4

    optimizer = torch.optim.SGD(second_model.parameters(), lr=learning_rate, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

    x_train = Glove_Data(glove_train_input, train_label, train_cap_bit)
    train_loader = data.DataLoader(dataset=x_train, batch_size=batch_size, shuffle=True)
    second_model.cuda()

    best_running_loss = 10_000
    running_loss = 0

    epochs = 50
    for e in range(epochs):
        for i , (x, y, l, c) in enumerate(train_loader):
            # test = x.reshape(batch_size,20,300).to(device)
            x = x.to(device)
            y = pack_padded_sequence(y, l, batch_first=True, enforce_sorted=False)
            y, _ = pad_packed_sequence(y, batch_first=True)
            y = y.to(device)
            c = c.to(device)
            
            output = second_model(x, l, c)
            output = torch.permute(output, (0,2,1))

            loss = loss_fn(output, y)

            running_loss += loss.item() / x.size(0)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()
        
        if (running_loss < best_running_loss):
            best_running_loss = running_loss
            stagnate = 0
        else:
            stagnate += 1
        
        if stagnate > 2:
            break

        if e % 2 == 0:
            print(f"Epoch: {e} || Loss: {running_loss}")
        running_loss = 0
    torch.save(second_model, 'blstm2.pt')
elif mode == 0:
    print("="*10, "Loading blstm2.pt", "="*10)
    # second_model = second_model.load_state_dict(torch.load('blstm2.pt'))
    second_model = torch.load('blstm2.pt')

x_train = Glove_Data(glove_train_input, train_label, train_cap_bit)
train_loader = data.DataLoader(dataset=x_train, batch_size=1)
train_pred, train_actual = glove_predict(second_model, device, train_loader, verbose=True)
train_output = back_to_tags(train_pred, ner_tag_dict)

write_prediction_perl(train_output, train_input_raw, train_label_raw, "train_prediction2.txt")

flat_x, flat_y = [], []
for output, label in zip(train_pred, train_actual):
    for i in range(len(output)):
        flat_x.append(output[i])
        flat_y.append(label[i])

#204567
print(metrics.classification_report(flat_y, flat_x))

x_dev = Glove_Data(glove_dev_input, dev_label, dev_cap_bit)
dev_loader = data.DataLoader(dataset=x_dev, batch_size=1)
dev_pred, dev_actual = glove_predict(second_model, device, dev_loader, verbose=True)
dev_output = back_to_tags(dev_pred, ner_tag_dict)

write_prediction_perl(dev_output, dev_label_raw, dev_label_raw, "dev_prediction2.txt")
write_prediction(dev_output, dev_input_raw, 'dev2.out')

flat_x, flat_y = [], []
for output, label in zip(dev_pred, dev_actual):
    for i in range(len(output)):
        flat_x.append(output[i])
        flat_y.append(label[i])

#204567
print(metrics.classification_report(flat_y, flat_x))

x_test = Glove_Data(glove_test_input, test_label, test_cap_bit)
test_loader = data.DataLoader(dataset=x_test, batch_size=1)
test_pred, test_actual = glove_predict(second_model, device, test_loader, verbose=True)
test_output = back_to_tags(test_pred, ner_tag_dict)

write_prediction(test_output, test_input_raw, 'test2.out')

new_glove_vocab = {}
new_glove_vocab_lookup = {}
new_glove_embeddings = []

new_glove_vocab['<unk>'] = 0
new_glove_vocab_lookup[0] = '<unk>'
new_glove_embeddings.append([0 for i in range(100)])


with gzip.open(glove, "rt", encoding='utf-8') as g_file:
    i = 1
    for line in g_file:
        temp = line.split()
        word = temp[0]
        features = temp[1:]
        new_glove_vocab[word] = i
        new_glove_vocab_lookup[i] = word
        new_glove_embeddings.append(list(map(float, features)))
        i+=1
        if word[0].isalpha():
            cap_word = word[0].upper() + word[1:]
            new_glove_vocab[cap_word] = i
            new_glove_vocab_lookup[i] = cap_word
            new_glove_embeddings.append(list(map(float, features)))
            i+=1

n = len(glove_vocab)
for i in range(n, n*2):
    index = i - n
    new_word = glove_vocab_lookup[index][0].upper() + glove_vocab_lookup[index][1:]
    glove_vocab_lookup[i] = new_word
    glove_vocab[new_word] = i

char_set = set()
char_set.add(' ')
for key in new_glove_vocab.keys():
    for char in key:
        char_set.add(char)

char_vocab = {}
char_vocab_lookup = {}

char_vocab['<pad>'] = 0
char_vocab_lookup[0] = '<pad>'

char_vocab['<unk>'] = 1
char_vocab_lookup[1] = '<unk>'

char_vocab['<start>'] = 2
char_vocab_lookup[2] = '<start>'

char_vocab['<end>'] = 3
char_vocab_lookup[3] = '<end>'

i = 4
for char in char_set:
    char_vocab[char] = i
    char_vocab_lookup[i] = char
    i += 1

char_vocab_len = len(char_vocab)

max_word_len = 0
for word in glove_vocab.keys():
    if len(word) > max_word_len:
        max_word_len = len(word)
max_word_len += 2

new_glove_train_input = new_glove_convert_to_idx(train_input_raw, new_glove_vocab)
new_glove_dev_input = new_glove_convert_to_idx(dev_input_raw, new_glove_vocab)
new_glove_test_input = new_glove_convert_to_idx(test_input_raw, new_glove_vocab)

batch_size = 3
x_train = BLSTM_CNN_Data(new_glove_train_input, train_label, train_cap_bit, char_vocab, max_word_len, new_glove_vocab_lookup)
train_loader = data.DataLoader(dataset=x_train, batch_size=batch_size, shuffle=False)

class CNN(torch.nn.Module):
    def __init__(self, char_vocab_size, char_embedding_size, channel_out1, kernel1, channel_out2, kernel2):
        super(CNN, self).__init__()
        
        self.char_embedding = torch.nn.Embedding(char_vocab_size, char_embedding_size)
        self.conv = torch.nn.Conv1d(char_embedding_size, channel_out1, kernel1, padding=1)
        self.relu = torch.nn.ReLU()
        self.maxpool = torch.nn.MaxPool1d(kernel_size=kernel1, stride=1, padding=1)
        self.conv2 = torch.nn.Conv1d(channel_out1, channel_out2, kernel2, padding=1)
        self.relu2 = torch.nn.ReLU()
        self.maxpool2 = torch.nn.MaxPool1d(kernel_size=kernel2, stride=1, padding=1)

    def forward(self, chars):
        embed = self.char_embedding(chars)
        output = embed.transpose(1, 2)
        output = self.conv(output)
        output = self.relu(output)
        output = self.maxpool(output)
        output = self.conv2(output)
        output = self.relu2(output)
        output = self.maxpool2(output)

        output, _ = torch.max(output, dim=2)
        return output

class BLSTM_CNN(torch.nn.Module):
    def __init__(self, word_embedding_features, embedding_dim, hidden_dim, linear_dim, D_out, dropout):
        super(BLSTM_CNN, self).__init__()

        self.cnn = CNN(char_vocab_len, char_embedding_size=30, channel_out1=60, kernel1=5, channel_out2=30, kernel2=3)

        self.embed = torch.nn.Embedding.from_pretrained(word_embedding_features)
        self.lstm = torch.nn.LSTM(embedding_dim+31, hidden_dim, bidirectional=True, batch_first=True)
        self.d = torch.nn.Dropout(p=dropout)
        self.linear = torch.nn.Linear(hidden_dim*2, linear_dim)
        self.elu = torch.nn.ELU()
        self.classifier = torch.nn.Linear(linear_dim, D_out, dtype=torch.float32)
        self.weight_init()

    def forward(self, text, x_char, lengths, cap):
        output = self.embed(text)

        batch_size = x_char.shape[0]
        char_emb = []
        for i in range(batch_size):
            char_emb.append(self.cnn(x_char[i,:,:]))
        
        char_emb = torch.stack(char_emb)

        cap = cap.unsqueeze(2)
        output = torch.cat([output, char_emb, cap], dim=2)

        output = pack_padded_sequence(output, lengths, batch_first=True, enforce_sorted=False)
        output, (h, c) = self.lstm(output)
        output, x = pad_packed_sequence(output, batch_first=True)

        output = self.d(output)
        output = self.linear(output)
        output = self.elu(output)
        output = self.classifier(output)

        return output
    
    def weight_init(self):
        torch.nn.init.xavier_uniform_(self.linear.weight)
        torch.nn.init.xavier_uniform_(self.classifier.weight)

weights = []
for key, value in ner_tag_dict.items():
    weights.append(1-(tag_weights[value]/tag_total))

weights = torch.tensor(weights)
weights = weights.to('cuda')

bonus_model = BLSTM_CNN(torch.tensor(new_glove_embeddings), 100, 256, 128, ner_tag_len, 0.33)
if mode == 1:
    loss_fn = torch.nn.CrossEntropyLoss(weight=weights.float(), ignore_index=-1)
    learning_rate = 0.05 # best=0.05
    batch_size = 4 # best=4

    optimizer = torch.optim.SGD(bonus_model.parameters(), lr=learning_rate, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

    x_train = BLSTM_CNN_Data(new_glove_train_input, train_label, train_cap_bit, char_vocab, max_word_len, new_glove_vocab_lookup)
    train_loader = data.DataLoader(dataset=x_train, batch_size=batch_size, shuffle=True)
    bonus_model.cuda()

    best_running_loss = 10_000
    running_loss = 0

    epochs = 50
    for e in range(epochs):
        for i , (x, y, l, c, x_char) in enumerate(train_loader):
            # test = x.reshape(batch_size,20,300).to(device)
            x = x.to(device)
            x_char = x_char.to(device)
            y = pack_padded_sequence(y, l, batch_first=True, enforce_sorted=False)
            y, _ = pad_packed_sequence(y, batch_first=True)
            y = y.to(device)
            c = c.to(device)
            
            output = bonus_model(x, x_char, l, c)
            output = torch.permute(output, (0,2,1))

            loss = loss_fn(output, y)

            running_loss += loss.item() / x.size(0)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        scheduler.step()
        if (running_loss < best_running_loss):
            best_running_loss = running_loss
            stagnate = 0
        else:
            stagnate += 1
        
        if stagnate > 2:
            break

        if e % 2 == 0:
            print(f"Epoch: {e} || Loss: {running_loss}")
        running_loss = 0

    x_train = BLSTM_CNN_Data(new_glove_train_input, train_label, train_cap_bit, char_vocab, max_word_len, new_glove_vocab_lookup)
    train_loader = data.DataLoader(dataset=x_train, batch_size=1, shuffle=False)
    train_pred, train_actual = new_glove_predict(bonus_model, device, train_loader, verbose=True)
    train_output = back_to_tags(train_pred, ner_tag_dict)

    write_prediction_perl(train_output, train_input_raw, train_label_raw, "train_prediction_bonus.txt")

    flat_x, flat_y = [], []
    for output, label in zip(train_pred, train_actual):
        for i in range(len(output)):
            flat_x.append(output[i])
            flat_y.append(label[i])

    #204567
    print(metrics.classification_report(flat_y, flat_x))

    x_dev = BLSTM_CNN_Data(new_glove_dev_input, dev_label, dev_cap_bit, char_vocab, max_word_len, new_glove_vocab_lookup)
    dev_loader = data.DataLoader(dataset=x_dev, batch_size=1, shuffle=False)
    dev_pred, dev_actual = new_glove_predict(bonus_model, device, dev_loader, verbose=True)
    dev_output = back_to_tags(dev_pred, ner_tag_dict)

    write_prediction_perl(dev_output, dev_input_raw, dev_label_raw, "dev_prediction_bonus.txt")

    flat_x, flat_y = [], []
    for output, label in zip(dev_pred, dev_actual):
        for i in range(len(output)):
            flat_x.append(output[i])
            flat_y.append(label[i])

    #204567
    print(metrics.classification_report(flat_y, flat_x))

    x_test = BLSTM_CNN_Data(new_glove_test_input, test_label, test_cap_bit, char_vocab, max_word_len, new_glove_vocab_lookup)
    test_loader = data.DataLoader(dataset=x_test, batch_size=1, shuffle=False)
    test_pred, test_actual = new_glove_predict(bonus_model, device, test_loader, verbose=True)
    test_output = back_to_tags(test_pred, ner_tag_dict)

    write_prediction(test_output, test_input_raw, 'pred.out')