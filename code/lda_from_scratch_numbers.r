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

## assign word IDs to each unique word
vocab <- unique(unlist(docs))

## replace words in documents with IDs
for (i in 1:length(docs))
  docs[[i]] <- match(docs[[i]], vocab)

print(docs)
