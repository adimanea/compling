## Generate a corpus
rawdocs <- c('eat turkey on turkey day holiday',
			 'i like to eat cake on holiday',
			 'turkey trot race on thanksgiving holiday',
			 'snail race the turtle',
			 'time travel space race',
			 'movie on thanksgiving',
			 'movie at air and space museum is cool movie',
			 'aspiring movie star')

## Generate list of documents
docs <- strsplit(rawdocs, split=' ', perl=T)

## LDA parameters
K <- 2          # number of topics
alpha <- 1      # hyperparameter; higher => scattered docs
eta <- .001     # hyperparameter
iterations <- 3 # for collaps Gibbs sampling (see doc)

print(docs)
