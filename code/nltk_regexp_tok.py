from nltk.tokenize import RegexpTokenizer

sentence = """Thomas Jefferson began building the Monticello at the
age of 26."""

tokenizer = RegexpTokenizer(r'\w+|$[0-9.]+|\S+')

print(tokenizer.tokenize(sentence))
