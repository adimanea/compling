import re

sentence = """Thomas Jefferson began building the Monticello at the\
age of 26."""

# prepare the regex beforehand
pattern = re.compile(r"([-\s.,;~?])+")

# apply the regex
tokens = pattern.split(sentence)

# filter out whitespace and punctuation
good_toks = [x for x in tokens if x and x not in '- \t\n.,;~?']

print(good_toks)
