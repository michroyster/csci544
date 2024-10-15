import pandas as pd
import numpy as np
import json

import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('data/train', delimiter='\t', header=None, names=['index', 'word', 'pos_tag'])

""" Task 1: Vocabulary Creation """
threshhold = 5 # tune this

vocabulary = df.groupby(["word"]).size().reset_index(name="occurences") # for grouping if needed. which to group by? index, pos_tag ??
vocabulary.sort_values(by='occurences', inplace=True, ascending=False)
vocabulary.reset_index(drop=True, inplace=True)

before = vocabulary.shape[0]

df = df.merge(vocabulary, how='left', on='word')
remainder = df.query("occurences < @threshhold").shape[0]

vocabulary_reduced = vocabulary.query("occurences >= @threshhold")
vocabulary_reduced.reset_index(drop=True, inplace=True)
vocabulary_reduced = pd.concat([pd.DataFrame({"word" : "<unk>", "occurences": remainder}, index=[0]), vocabulary_reduced]).reset_index(drop=True)

after = vocabulary_reduced.shape[0]
vocabulary_reduced.insert(loc=1, column='index', value=vocabulary_reduced.index)

order = vocabulary_reduced['word'].to_list()
vocab = dict()
for i in range(len(order)):
    vocab[i] = order[i]

print(f"Vocab Before: {before}\nAfter:  {after}") # Output to task 1
vocabulary_reduced.to_csv("vocab.txt", sep='\t', index=False) # Output to task 1
df['word'] = np.where(df['occurences'] < threshhold, '<unk>', df['word'])


""" Task 2: Model Learning """
df['pos_shift'] = df['pos_tag'].shift(-1)

""" Learn transition table - the probability of transitioning from one POS to another POS 
    |S| X |S|
"""
transitions = df.groupby('pos_tag')['pos_shift'].value_counts().unstack(fill_value=0)
transitions['ROW_TOTAL'] = transitions.iloc[:,0:].sum(axis=1)
transitions.iloc[:,0:-1] = transitions.iloc[:,0:].div(transitions['ROW_TOTAL'], axis=0)
transitions.drop(columns=['ROW_TOTAL'], inplace=True)
transitions.sort_index(inplace=True)
transitions = transitions.reindex(sorted(transitions.columns), axis=1)

initial = df.query("index == 1")#.groupby('pos_tag')['word'].value_counts().unstack(fill_value=0)
initial = initial.assign(pos_tag='<s>')
initial = initial.groupby('pos_tag')['pos_shift'].value_counts().unstack(fill_value=0)
initial['ROW_TOTAL'] = initial.iloc[:,0:].sum(axis=1)
initial.iloc[:,0:-1] = initial.iloc[:,0:].div(initial['ROW_TOTAL'], axis=0)
initial.drop(columns=['ROW_TOTAL'], inplace=True)

difference = set(transitions.index.to_list()) - set(initial.columns)
for d in difference:
    initial[d] = 0

initial = initial.reindex(sorted(initial.columns), axis=1)

""" Learn emission table - the probability of observing a given word at a given state
    |V| X |S|
"""
emission = df.groupby('pos_tag')['word'].value_counts().unstack(fill_value=0)

emission['POS_ROW_TOTAL'] = emission.iloc[:,0:].sum(axis=1)
emission.iloc[:, 0:-1] = emission.iloc[:,0:].div(emission['POS_ROW_TOTAL'], axis=0)
emission.drop(columns=['POS_ROW_TOTAL'], inplace=True)
emission.sort_index(inplace=True)
emission = emission.reindex(order, axis=1)

# Create Json as output for task 2
t = dict()
e = dict()
t_json = dict()
e_json = dict()

for c in transitions.columns:
    for r in transitions.index.to_list():
        if transitions.loc[r][c] != 0:
            t[(c,r)] = transitions.loc[r][c]
            t_json[str((c,r))] = transitions.loc[r][c]

for c in emission.columns:
    for r in emission.index.to_list():
        if emission.loc[r][c] != 0:
            e[(c,r)] = emission.loc[r][c]
            e_json[str((c,r))] = emission.loc[r][c]

print(f"Parameters in transition: {len(t.keys())}") # output for task 2
print(f"Parameters in emission  : {len(e.keys())}") # output for task 2

""" Put t and e tables into json """
model_data = {'transitions': t_json, 'emissions': e_json}
with open('hmm.json', 'w') as f:
    json.dump(model_data, f)


""" Creating unique indexes for pos_tags and vocab words """
unique_tags = sorted(df['pos_tag'].unique())
pos_tags = dict()
for i in range(len(unique_tags)):
    pos_tags[i] = unique_tags[i]

tag_index = dict()
vocab_index = dict()
for key in pos_tags:
    tag_index[pos_tags[key]] = key
for key in vocab:
    vocab_index[vocab[key]] = key

""" Converting dataframes to numpy arrays """
pi = initial.values
e = emission.values
t = transitions.values

""" pos_tags : unique index across emission (rows), transition (rows and cols) and initial (cols)
    tag_index: Get the index given a tag
    vocab    : unique index for each word in emission
    vocab_index: get the index given a word
"""

""" Task 3: Greedy Decoding with HMM """
def load_batch(filename):
    sentences = []
    with open(filename, 'r') as file:
        s = []
        for i, line in enumerate(file):
            values = line.split()
            if values:
                s.append(values[1])
            else:
                sentences.append(s)
                s = []
        sentences.append(s)
        return sentences


test_data = load_batch('data/test')
dev_data = load_batch('data/dev')

def greedy_decode(o, t, e, pi, pos_tags, vocab_index):
    prediction = []

    e_prob = e[:,vocab_index.get(o[0], 0)]

    first = np.argmax(np.multiply(pi, e_prob))
    prediction.append(pos_tags[first])
    prev = first
    for i in range(1,len(o)):
        e_prob =e[:,vocab_index.get(o[i], 0)]
        position = np.argmax(np.multiply(t[prev, :], e_prob))
        prediction.append(pos_tags[position])
        prev = position
    return prediction

dev_results = []
for sentence in dev_data:
    pred = greedy_decode(sentence, t, e, pi, pos_tags, vocab_index)
    for word, prediction in zip(sentence, pred):
        dev_results.append(prediction)

test_results = []
for sentence in test_data:
    pred = greedy_decode(sentence, t, e, pi, pos_tags, vocab_index)
    i=1
    for word, prediction in zip(sentence, pred):
        test_results.append(f"{i}\t{word}\t{prediction}")
        i+=1
    test_results.append("")
    i=1

# Task 3 Output
with open('greedy.out', 'w') as file:
    for result in test_results:
        file.write(f"{result}\n")

answers = []
with open('data/dev', 'r') as file:
    for line in file:
        values = line.split()
        if values:
            answers.append(values[2])


total, correct = 0, 0
for ans, pred in zip(answers, dev_results):
    if ans == pred:
        correct += 1
    total += 1

# Task 3 Output
print(f"Greedy Dev Data Accuracy: {np.round(correct/total,4)}")

""" Task 4: Viterbi Decoding with HMM """
def viterbi(o, t, e, pi, pos_tags, vocab_index):
    obs = len(o) # number of observations
    tags = len(t) # number of pos_tags

    opt = np.zeros((tags, obs), dtype=np.longdouble)
    backtrack = np.zeros((tags, obs))

    e_prob = e[:,vocab_index.get(o[0], 0)]
    opt[:,0] = np.multiply(pi, e_prob)

    for i in range(1, obs):
        for j in range(tags):
            # Calculate probabilities for position
            # Get the max probability for position
            # Get the position of the max probability
            probs = opt[:,i-1] * t[:,j] * e[j,vocab_index.get(o[i], 0)]
            opt[j, i] = np.max(probs)
            backtrack[j, i] = np.argmax(probs)

    # Iterate backwards to get the path
    last = np.argmax(opt[:,obs-1])
    predictions = [last]

    for i in range(2, obs+1):
        predictions.insert(0, int(backtrack[int(predictions[0])][-i+1]))

    tag_prediction = [pos_tags[predictions[i]] for i in range(len(predictions))]
    return tag_prediction

dev_results = []
for sentence in dev_data:
    pred = viterbi(sentence, t, e, pi, pos_tags, vocab_index)
    for word, prediction in zip(sentence, pred):
        dev_results.append(prediction)

test_results = []
for sentence in test_data:
    pred = viterbi(sentence, t, e, pi, pos_tags, vocab_index)
    i=1
    for word, prediction in zip(sentence, pred):
        test_results.append(f"{i}\t{word}\t{prediction}")
        i+=1
    test_results.append("")
    i=1

# Task 4 Output
with open('viterbi.out', 'w') as file:
    for result in test_results:
        file.write(f"{result}\n")

answers = []
with open('data/dev', 'r') as file:
    for line in file:
        values = line.split()
        if values:
            answers.append(values[2])


total, correct = 0, 0
for ans, pred in zip(answers, dev_results):
    if ans == pred:
        correct += 1
    total += 1

# Task 4 Output
print(f"Viterbi Dev Data Accuracy: {np.round(correct/total,4)}")