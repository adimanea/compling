from nlpia.book.examples.ch04_catdog_lsa_sorted import lsa_models, prettify_tdm

bow_svd, tfidf_svd = lsa_models()
tdm = bow_svd['tdm']

import numpy as np
U, s, Vt = np.linalg.svd(tdm)

import pandas as pd
print("> Diagonal form of S matrix:")
print(s.round(1))

S_full = np.zeros((len(U), len(Vt)))
pd.np.fill_diagonal(S_full, s)
print("> Full form of S matrix:")
print(pd.DataFrame(S_full).round(1))
