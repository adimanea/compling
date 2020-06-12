# FILE: simple_tok.py
#############################################################
# This file should be read and used in conjuction with the
# notes contained in the notes.org file in the current repo.
# Explanations and comments are found there. This is merely
# an exported standalone version to be interpreted outside
# the notes.
#############################################################

import numpy as np

sentence = """Thomas Jefferson began building Monticello
			  at the age of 26."""

token_sequence = str.split(sentence)
vocab = sorted(set(token_sequence))
', '.join(vocab)
num_tokens = len(token_sequence)
vocab_size = len(vocab)
onehot_vectors = np.zeros((num_tokens, vocab_size), int)
for i, word in enumerate(token_sequence):
	onehot_vectors[i, vocab.index(word)] = 1
' '.join(vocab)
print(onehot_vectors)
