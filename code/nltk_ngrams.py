import re
from nltk.util import ngrams

sentence = """Thomas Jefferson began building Monticello at the
age of 26."""

# get the tokens (words) first with a simple split
pattern = re.compile(r"([-\s.,;~?])+")
tokens = pattern.split(sentence)
# disregard whitespace and punctuation
tokens = [x for x in tokens if x and x not in '- \t\n.,;~?']

# N-grams with NLTK
print("2- and 3-grams as tuples:")
print(list(ngrams(tokens, 2)))
print(list(ngrams(tokens, 3)))

print("2-grams joined with whitespace:")
two_grams = list(ngrams(tokens, 2))
print([" ".join(x) for x in two_grams])
