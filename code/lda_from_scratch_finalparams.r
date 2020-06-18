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
  for (w in 1:length(docs[[d]])) {          # for each token in document d
	ta[[d]][w] <- sample(1:K, 1)            # randomly assign topic to token w
	ti <- ta[[d]][w]                        # topic index
	wi <- docs[[d]][w]                      # word ID for token w
	wt[ti,wi] <- wt[ti,wi] + 1              # update word-topic count matrix
  }
}

dt <- matrix(0, length(docs), K)
for (d in 1:length(docs)) {                 # for each document d
  for (t in 1:K) {                          # for each topic t
	dt[d,t] <- sum(ta[[d]]==t)              # count tokens in document d assigned to topic t
  }
}

for (i in 1:iterations) {                   # for each pass through the corpus
  for (d in 1:length(docs)) {               # for each document
	for (w in 1:length(docs[[d]])) {        # for each token
	  t0 <- ta[[d]][w]                      # initial topic assignment to token w
	  wid <- docs[[d]][w]                   # word ID of token w
	  dt[d,t0] <- dt[d,t0] - 1              # don't include token w in the DT-matrix when sampling for w
	  wt[t0,wid] <- wt[t0,wid] - 1          # don't include toekn w in WT-matrix when sampling for w

	  ## UPDATE TOPIC ASSIGNMENT FOR EACH WORD: COLLAPSED GIBBS SAMPLING MAGIC
	  denom_a <- sum(dt[d,]) + K + alpha    # no. tokens in doc + no topics * alpha
	  denom_b <- rowSums(wt) + length(vocab) * eta # no. tokens for topic + no words in vocab * eta
	  p_z <- (wt[,wid] + eta) / denom_b * (dt[d,] + alpha) / denom_a    # conditional probability
	  t1 <- sample(1:K, 1, prob=p_z/sum(p_z))   # draw topic for each word n from multinomial computed above

	  ta[[d]][w] <- t1                      # update topic assignment list with new sampled topic for w
	  dt[d,t1] <- dt[d,t1] + 1              # re-increment DT matrix with new topic assignment for w
	  wt[t1,wid] <- wt[t1,wid]+1            # ------"----- WT -------"----------------"--------------
	}
  }
}

######################################################################

## topic probabilities per document
theta <- (dt + alpha) / rowSums(dt + alpha)
print(theta)

## topic probabilities per word
phi <- (wt + eta) / (rowSums(wt+eta))
colnames(phi) <- vocab
print(phi)
