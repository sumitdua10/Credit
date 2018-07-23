import pandas as pd
import numpy as np

x = np.random.randint(10,100,(5,4))
print(x)
df = pd.DataFrame(x)
df.columns = ['a','b','c','d']
df.loc[3,'a'] = '..'
df.loc[3,'b'] = '..'
print(df)
print(df.dtypes)
for col in df.columns:
   for i,j in enumerate(df[col]):
    #print(j)
        if not str(j).isnumeric():
            print("Col, ", col, "index ", i,"Value ",j)
            df.loc[i,col] = np.nan
print(df)
#df['a'.re]
#print(df.drop('..'))

