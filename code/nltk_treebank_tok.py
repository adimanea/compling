from nltk.tokenize import TreebankWordTokenizer

sentence = """Monticello wasn't designed as UNESCO World Heritage\
Site until 1987."""

tokenizer = TreebankWordTokenizer()
print(tokenizer.tokenize(sentence))
