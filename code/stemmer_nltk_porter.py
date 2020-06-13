from nltk.stem.porter import PorterStemmer

stemmer = PorterStemmer()
print("Porter Stemmer for 'Dish washer's washed dishes' gives:")
print(' '.join([stemmer.stem(w).strip("'") for w in
				"Dish washer's washed dishes".split()]))
