#!/usr/bin/env python

import pandas as pd

# initialize a dataframe
df = pd.DataFrame(
    [[21, 72, 67],
     [23, 78, 69],
     [32, 74, 56],
     [52, 54, 76]],
    columns=['a', 'b', 'c'])

print('DataFrame\n----------\n', df)

# convert dataframe to numpy array
arr = df.values

print('\nNumpy Array\n----------\n', arr)

arr2 = df.as_matrix(['a', 'b'])
print('\nNumpy Array2\n----------\n', arr2)
