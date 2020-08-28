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

students = [('jack', 34, 'Sydeny'),
            ('Riti', 30, 'Delhi'),
            ('Andi', 16, 'New York'),
            ('Hias', 1, 'New Flork'),
            ('Hans', 22, 'New Dork'),
            ('Franz', 39, 'Old Yorl')]

dfObj = pd.DataFrame(students, columns=['Name', 'Age', 'City'])

print(dfObj)

# https://thispointer.com/select-rows-columns-by-name-or-index-in-dataframe-using-loc-iloc-python-pandas/
print('select just some rows: ')
rowData = dfObj.loc[[0, 2], :]
print(rowData)
