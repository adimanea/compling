from collections import Counter

from nltk.tokenize import TreebankWordTokenizer
tokenizer = TreebankWordTokenizer()

from nlpia.data.loaders import kite_text

tokens = tokenizer.tokenize(kite_text.lower())
token_counts = Counter(tokens)

import nltk
nltk.download('stopwords', quiet=True)
stopwords = nltk.corpus.stopwords.words('english')

tokens = [x for x in tokens if x not in stopwords]
# word counts from the article, that are NOT stopwords
kite_counts = Counter(tokens)

document_vector = []
doc_length = len(tokens)
# frequency of each word = apparition count / doc length
for key, value in kite_counts.most_common():
	document_vector.append(value / doc_length)

print("The first 5 frequencies are:")
print(document_vector[:5])
