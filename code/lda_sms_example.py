import pandas as pd
from nlpia.data.loaders import get_data
pd.options.display.width = 120

sms = get_data('sms-spam')                      # corpus

index = ['sms{}{}'.format(i, '!'*j) for (i, j) in\
		 zip(range(len(sms)), sms.spam)]
sms = pd.DataFrame(sms.values, columns=sms.columns, index=index)
sms['spam'] = sms.spam.astype(int)
print("You've got " + str(len(sms)) + " messages in total")
print("Of which " + str(sms.spam.sum()) + " are spam")
print("Here is an example listing:")
print(sms.head(6))

# tokenization and TF-IDF for the SMSs
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize.casual import casual_tokenize
tfidf_mode = TfidfVectorizer(tokenizer=casual_tokenize)
tfidf_docs = tfidf_mode.fit_transform(raw_documents=sms.text).toarray()
print("--------------------------------------------------")
print("The tokenizer gives the following (texts, tokens) numbers: " + \
	  str(tfidf_docs.shape))

mask = sms.spam.astype(bool).values     # select only the spam items
spam_centroid = tfidf_docs[mask].mean(axis=0)
ham_centroid = tfidf_docs[~mask].mean(axis=0)

print("--------------------------------------------------")
print("Here is a part of the centroid for SPAM messages:")
print(spam_centroid[:5].round(2))
print("Here is a part of the centroid for HAM messages:")
print(ham_centroid[:5].round(2))

# subtract one centroid from the other to get the line between them
spamminess_score = tfidf_docs.dot(spam_centroid - ham_centroid)
print("--------------------------------------------------")
print("A part of the line between the centroids is:")
print(spamminess_score[:5].round(2))

# assign scores (like probabilities)
from sklearn.preprocessing import MinMaxScaler
sms['lda_score'] = MinMaxScaler().fit_transform(spamminess_score.reshape(-1,1))
sms['lda_predict'] = (sms.lda_score > .5).astype(int)

print("--------------------------------------------------")
print("Some listings:")
print(sms['spam lda_predict lda_score'.split()].round(2).head(6))

# false positives and false negatives
from pugnlp.stats import Confusion
print("--------------------------------------------------")
print("False positives and false negatives:")
print(Confusion(sms['spam lda_predict'.split()]))
