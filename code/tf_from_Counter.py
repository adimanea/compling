import sys
sys.path.append("./code")

import bagofwords_count as bow

times_harry_appears = bow.bag_of_words['harry']
num_unique_words = len(bow.bag_of_words)
tf = times_harry_appears / num_unique_words
tf_harry = round(tf, 4)
print("TF for 'harry' is " + str(tf_harry))
