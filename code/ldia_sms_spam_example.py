# HACK: ignore FutureWarnings (related to Pandas)
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# take SMS data
import numpy as np
import pandas as pd
from nlpia.data.loaders import get_data

pd.options.display.width = 120          # better printing of DataFrame
sms = get_data('sms-spam')

# add '!' to spam messages for easy spotting
index = ['sms{}{}'.format(i, '!'*j)
		 for (i, j) in zip(range(len(sms)), sms.spam)]
sms.index = index

######################################################################
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import casual_tokenize

np.random.seed(42)

# compute BOW vectors in scikit-learn
counter = CountVectorizer(tokenizer=casual_tokenize)
bow_docs = pd.DataFrame(counter.fit_transform(raw_documents=sms.text).toarray(),
						index=index)
column_nums, terms = zip(*sorted(zip(counter.vocabulary_.values(),
					 counter.vocabulary_.keys())))
bow_docs.columns = terms

# check
print(sms.loc['sms0'].text)
print("> BOW:")
print(bow_docs.loc['sms0'][bow_docs.loc['sms0'] > 0].head())

# LDiA
from sklearn.decomposition import LatentDirichletAllocation as LDiA

ldia = LDiA(n_components=16, learning_method='batch')
ldia = ldia.fit(bow_docs)
print("> LDiA size (topics, words): " + str(ldia.components_.shape))

# let's now use PCA
from sklearn.decomposition import PCA

from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize.casual import casual_tokenize

tfidf = TfidfVectorizer(tokenizer=casual_tokenize)
tfidf_docs = tfidf.fit_transform(raw_documents=sms.text).toarray()

pca = PCA(n_components=16)
pca = pca.fit(tfidf_docs)
pca_topic_vectors = pca.transform(tfidf_docs)
columns = ['topic{}'.format(i) for i in range(pca.n_components)]
pca_topic_vectors = pd.DataFrame(pca_topic_vectors,
				 columns=columns, index=index)

pd.set_option('display.width', 75)
components = pd.DataFrame(ldia.components_.T, index=terms, columns=columns)
print("> Some terms and topics:")
print(components.round(2).head(3))

print("> Sorted important terms:")
print(components.topic3.sort_values(ascending=False)[:10])

# compute LDiA topic vectors
ldia16_topic_vectors = ldia.transform(bow_docs)
ldia16_topic_vectors = pd.DataFrame(ldia16_topic_vectors, index=index,
					columns=columns)
print("> LDiA topic vectors:")
print(ldia16_topic_vectors.round(2).head())

# LDiA + LDA check for spam analysis
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(ldia16_topic_vectors,
							sms.spam, test_size=0.5,
							random_state=271828)
lda = LDA(n_components=1)
lda = lda.fit(X_train, y_train)
sms['ldia16_spam'] = lda.predict(ldia16_topic_vectors)
print("> LDA + LDiA Spam Accuracy: " +
	  str(round(float(lda.score(X_test, y_test)), 2)))
