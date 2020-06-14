from nlpia.book.examples.ch04_catdog_lsa_sorted import lsa_models, prettify_tdm

bow_svd, tfidf_svd = lsa_models()
# the sparse, pretty print
print(prettify_tdm(**bow_svd))
# **arg unpacks a dictionary argument and feeds
# each key-value pair as an argument to the function called

tdm = bow_svd['tdm']
# the term-document matrix print
print(tdm)
