Author: Michael Royster

Description: 
	Training on the data set in data/train, this program creates a vocabulary and
	replaces words under a given thresshold with token <unk>. Then, emission and transition
	probabilities are calculated.

Files Generated:
	vocab.txt - 	words in vocabulary with index and number of occurrences
	hmm.json - 	two dictionaries containing transition and emission probabilities
	greedy.out - 	predictions on the data/test data set utilizing greedy algorithm
	viterbi.out - 	predictions on the data/test data set utilizing viterbi algorithm

Dependencies:
	Python 3.9.13
	numpy 1.23.3
	pandas 1.5.2
	json 2.0.9
	warnings - only used to supress annoying warnings

Run Instruction:
	- Program assumes there is a folder in local directory called data containing dev, test and train
	ex: data/dev, data/test, data/train

Command: python hw2_Michael_Royster.py
