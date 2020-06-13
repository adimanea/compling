import nltk

nltk.download('stopwords')
stop_words = nltk.corpus.stopwords.words('english')
print("There are " + str(len(stop_words)) + " stopwords in NLTK English DB")

print("Some of them are:")
print(stop_words[:10])

print("The shortest are:")
print([sw for sw in stop_words if len(sw) == 1])
