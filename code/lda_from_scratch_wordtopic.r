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

## assign word IDs to each unique word
vocab <- unique(unlist(docs))

## replace words in documents with IDs
for (i in 1:length(docs))
  docs[[i]] <- match(docs[[i]], vocab)


## 1. Randomly assign topics to words in each doc.
## 2. Generate word-topic count matrix
wt <- matrix(0, K, length(vocab))           # initialize wt matrix
ta <- sapply(docs, function(x) rep(0, length(x)))   # initialize topic assignment list
for (d in 1:length(docs)) {                 # for each document
  for (w in 1:length(docs[[d]])) {           # for each token in document d
	ta[[d]][w] <- sample(1:K, 1)            # randomly assign topic to token w
	ti <- ta[[d]][w]                        # topic index
	wi <- docs[[d]][w]                      # word ID for token w
	wt[ti,wi] <- wt[ti,wi] + 1              # update word-topic count matrix
  }
}

print(wt)               # word-topic count matrix
print(ta)               # token-topic assignment list
