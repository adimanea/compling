import nltk
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()
adjectives = ["better", "good", "goods", "best"]
nouns = ["better", "goods", "goodness"]

print("> We lemmatize the following adjectives:")
for adj in adjectives:
	print(adj, end=" ")
print("\n> And get respectively:")
for adj in adjectives:
	print(lemmatizer.lemmatize(adj, pos="a"), end=" ")

print("\n> Now the following nouns:")
for n in nouns:
	print(n, end=" ")
print("\n> And get respectively:")
for n in nouns:
	print(lemmatizer.lemmatize(n, pos="n"), end=" ")
