from nlpia.book.examples.ch04_catdog_lsa_sorted import lsa_models, prettify_tdm

bow_svd, tfidf_svd = lsa_models()
tdm = bow_svd['tdm']

import numpy as np
U, s, Vt = np.linalg.svd(tdm)

import pandas as pd
print(pd.DataFrame(Vt).round(2))
