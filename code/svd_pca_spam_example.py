import pandas as pd
from nlpia.data.loaders import get_data

pd.options.display.width = 120          # better printing of DataFrame
sms = get_data('sms-spam')

# add '!' to spam messages for easy spotting
index = ['sms{}{}'.format(i, '!'*j)
		 for (i, j) in zip(range(len(sms)), sms.spam)]
print("> Some messages:")
print(sms.head(6))

# calculate TF-IDF vectors for each message
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize.casual import casual_tokenize

tfidf = TfidfVectorizer(tokenizer=casual_tokenize)
tfidf_docs = tfidf.fit_transform(raw_documents=sms.text).toarray()
print("> Vocabulary for messages: " + str(len(tfidf.vocabulary_)))

tfidf_docs = pd.DataFrame(tfidf_docs)
# center the vectorized documents by subtracting the mean
tfidf_docs = tfidf_docs - tfidf_docs.mean()

print("> The array is now " + str(tfidf_docs.shape))
print("> There are " + str(sms.spam.sum()) + " messages marked as spam")

# let's now use PCA
from sklearn.decomposition import PCA

pca = PCA(n_components=16)
pca = pca.fit(tfidf_docs)
pca_topic_vectors = pca.transform(tfidf_docs)
columns = ['topic{}'.format(i) for i in range(pca.n_components)]
pca_topic_vectors = pd.DataFrame(pca_topic_vectors, \
								 columns=columns, index=index)
print("> PCA topic matrix (first 6):")
print(pca_topic_vectors.round(3).head(6))

# sort the vocabulary by term count
column_nums, terms = zip(*sorted(zip(tfidf.vocabulary_.values(),
									 tfidf.vocabulary_.keys())))

# now show the terms contained
weights = pd.DataFrame(pca.components_, columns=terms,
					   index = ['topic{}'.format(i) for i in range(16)])
pd.options.display.max_columns = 8
print(weights.head(4).round(3))

# some are uninteresting, so let's focus on some terms
pd.options.display.max_columns = 12
deals = weights['! ;) :) half off free crazy deal \
only $ 80 %'.split()].round(3) * 100

print(deals)

# let's see how many topics are about these "deals"
print(deals.T.sum())

# now use truncated SVD to keep just the 16 most interesting topics
from sklearn.decomposition import TruncatedSVD

svd = TruncatedSVD(n_components=16, n_iter=100)
svd_topic_vectors = svd.fit_transform(tfidf_docs.values)
svd_topic_vectors = pd.DataFrame(svd_topic_vectors, columns=columns,
								 index=index)
print("> SVD Topic Vectors:")
print(svd_topic_vectors.round(3).head(6))
# which are the same as PCA, given the n_iter = 100 (large)

# evaluate performance:
# compute the dot product for the first 6 topic vectors
# if cosine similarity is large wrt spam messages, it's OK
import numpy as np
svd_topic_vectors = (svd_topic_vectors.T /
					 np.linalg.norm(svd_topic_vectors, axis=1)).T
print("> Notice cosine similarity wrt spam messages:")
print(svd_topic_vectors.iloc[:10].dot(svd_topic_vectors.iloc[:10].T).round(1))
