from collections import OrderedDict, Counter
from nltk.tokenize import TreebankWordTokenizer
tokenizer = TreebankWordTokenizer()

from nlpia.data.loaders import kite_text, kite_history

kite_intro = kite_text.lower()
intro_tokens = tokenizer.tokenize(kite_intro)
kite_history = kite_history.lower()
history_tokens = tokenizer.tokenize(kite_history)

intro_total = len(intro_tokens)
print("The kite text contains " + str(intro_total) + " tokens")
history_total = len(history_tokens)
print("The kite history contains " + str(history_total) + " tokens")

intro_tf = {}
history_tf = {}
intro_counts = Counter(intro_tokens)
intro_tf['kite'] = intro_counts['kite'] / intro_total
history_counts = Counter(history_tokens)
history_tf['kite'] = history_counts['kite'] / history_total
print("Term frequency of 'kite' in intro is " + str(round(intro_tf['kite'], 4)))
print("Term frequency of 'kite' in history is " + str(round(history_tf['kite'], 4)))

# maybe the counts are not that relevant compared to 'and'
intro_tf['and'] = intro_counts['and'] / intro_total
history_tf['and'] = history_counts['and'] / history_total
print("Term frequency of 'and' in intro is " + str(round(intro_tf['and'], 4)))
print("Term frequency of 'and' in history is " + str(round(history_tf['and'], 4)))

# let's use rarity for IDF
num_docs_containing_and = 0
for doc in [intro_tokens, history_tokens]:
	if 'and' in doc:
		num_docs_containing_and += 1

num_docs_containing_kite = 0
for doc in [intro_tokens, history_tokens]:
	if 'kite' in doc:
		num_docs_containing_kite += 1

num_docs_containing_china = 0
for doc in [intro_tokens, history_tokens]:
	if 'china' in doc:
		num_docs_containing_china += 1

# TF of "China"
intro_tf['china'] = intro_counts['china'] / intro_total
history_tf['china'] = history_counts['china'] / history_total

# IDF
num_docs = 2
intro_idf = {}
history_idf = {}

intro_idf['and'] = num_docs / num_docs_containing_and
intro_idf['kite'] = num_docs / num_docs_containing_kite
intro_idf['china'] = num_docs / num_docs_containing_china

history_idf['and'] = num_docs / num_docs_containing_and
history_idf['kite'] = num_docs / num_docs_containing_kite
history_idf['china'] = num_docs / num_docs_containing_china

# TF-IDF
intro_tfidf = {}
intro_tfidf['and'] = intro_tf['and'] * intro_idf['and']
intro_tfidf['kite'] = intro_tf['kite'] * intro_idf['kite']
intro_tfidf['china'] = intro_tf['china'] * intro_idf['china']

history_tfidf = {}
history_tfidf['and'] = history_tf['and'] * history_idf['and']
history_tfidf['kite'] = history_tf['kite'] * history_idf['kite']
history_tfidf['china'] = history_tf['china'] * history_idf['china']

# example prints
print("TF-IDF for 'kite' in intro text is " + str(round(intro_tfidf['kite'], 4)))
print("TF-IDF for 'kite' in history text is " + str(round(history_tfidf['kite'], 4)))
