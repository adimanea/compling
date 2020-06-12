import sys
sys.path.append('~/repos/0-mine/compling/code/')

import simple_tok as st
import pandas as pd

print("Pandas print:")
print(pd.DataFrame(st.onehot_vectors, columns=st.vocab))

df = pd.DataFrame(st.onehot_vectors, columns=st.vocab)
df[df == 0] = ''                # ignore zeros
print("Prettier, no zeros:")
print(df)
