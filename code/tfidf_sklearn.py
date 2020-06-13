from sklearn.feature_extraction.text import TfidfVectorizer
from nlpia.data.loaders import kite_text

# case fold the text, but store it as a "text object"
# to be fed later (string not accepted)
corpus = [kite_text.lower()]

vectorizer = TfidfVectorizer(min_df=1)
model = vectorizer.fit_transform(corpus)
# convert the sparse matrix to a dense numpy-like version
print(model.todense().round(2))
