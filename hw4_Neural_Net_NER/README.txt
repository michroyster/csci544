Author: Michael Royster

Description:

This program was developed in jupyter notebootk hw4.ipynb and then adapted to the script: Royster_Michael_hw4.py

	The program will generate multiple files:
		1. blstm1.pt - trained model for task 1
		2. blstm2.pt - trained model for task 2
		3. dev1.out, dev2.out, test1.out and test2.out - dev and test predictions
		4. pred.out - Bonus predictions
	Additionally, these files will be generated and can be ignored as they are only created for
	testing against the provided perl grading script and can be ignored:
		* train_prediction1, train_prediction2
		* dev_prediction1, dev_prediction2

Dependencies:
	Python 3.9.13
	torch 1.13.1+cu116
	pandas 1.5.3
	sklearn 1.2.1
	numpy 1.24.2


Run Instruction:
	- program assumes files are present: data/train, data/dev and data/test
	- program also assumes that 'glove.6B.100d.gz' will be in same directory

Run command:
********Command line:
******* 1. To train the models AND generate the output:   python Royster_Michael_hw4.py train
******* 2. To predict based on the generated models only: python Royster_Michael_hw4.py predict
**************************************************************************************************
	OR you can run the jupyter notebook to both train and generate output files.


