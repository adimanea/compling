import re

sentence = """Thomas Jefferson began building Monticello at the\
age of 26."""

# regex to split at whitespace or punctuation
# that appears at least once
tokens = re.split(r'[-\s.,;!?]+', sentence)

print(tokens)
