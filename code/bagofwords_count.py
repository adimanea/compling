from nltk.tokenize import TreebankWordTokenizer
from collections import Counter

sentence = """The faster Harry got to the store, the faster
Harry, the faster, would get home."""

tokenizer = TreebankWordTokenizer()
tokens = tokenizer.tokenize(sentence.lower())
print("Individual tokens (words):")
for tok in tokens:
	print(tok, end=" ")

bag_of_words = Counter(tokens)
print("\nCleaned up now, with counts as a Counter:")
print(bag_of_words)

# collections.Counter objects have a bult-in method
# which prints the most common occurrences
print("The 3 most common words are:")
print(bag_of_words.most_common(3))
