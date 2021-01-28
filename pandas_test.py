#!/usr/bin/env python
# Software License Agreement (GNU GPLv3  License)
#
# Copyright (c) 2020, Roland Jung (roland.jung@aau.at) , AAU, KPK, NAV
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
#
# Requirements:
# numpy, matplotlib
########################################################################################################################
# !/usr/bin/env python
from sys import version_info
import pandas as pd
import numpy as np

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
if version_info[0] < 3:
    arr2 = df.as_matrix(['a', 'b'])
else:
    arr2 = df[['a', 'b']].to_numpy()
print('\nNumpy Array2\n----------\n', arr2)

students = [('jack', 34, 'Sydeny'),
            ('Riti', 30, 'Delhi'),
            ('Andi', 16, 'New York'),
            ('Sepp', 16, 'New Orleans'),
            ('Nancy', 16, 'New Orleans'),
            ('Hias', 1, 'New Flork'),
            ('Hans', 22, 'New Dork'),
            ('Franz', 39, 'Old Yorl')]

dfObj = pd.DataFrame(students, columns=['Name', 'Age', 'City'])

print(dfObj)

# https://thispointer.com/select-rows-columns-by-name-or-index-in-dataframe-using-loc-iloc-python-pandas/
print('select just some rows: ')
rowData = dfObj.loc[[0, 2], :]
print(rowData)

print('sort: ')
dfObj.sort_values(by=['Age'])
print(dfObj)

print('select just a column: ')
colData = dfObj.get(['Age'])
print(colData)
print('col values:')
colValues = colData.to_numpy()  # same as .values
print(colValues)
print('col values unique:')
colValues = np.unique(colValues)
print(colValues)

print('select just a column with condition: ')
colData = dfObj.loc[dfObj['Age'] > 16]
print(colData)

print('select just a column with condition: ')
colData = dfObj.loc[dfObj['Age'] == 16]
print(colData)

print('select just a column with multiple conditions: ')
colData = colData.loc[dfObj['City'] == 'New Orleans']
print(colData)

print('print cities: ')
print(colData['City'])

df = pd.DataFrame({'angles': [0, 3, 4],
                   'degrees': [360, 180, 360]})

print(df)
adegrees = df['degrees'].mean()
print('mean deg: ' + str(adegrees))

df2 = df + df

print(df2)

df2['adegrees'] = adegrees
print(df2)


arr1 = np.array([1, 2, 3, 4, 5, 6, 7])
idx = [2, 1, 0, 6, 4, 3, 5]
arr2 = arr1[idx]
print(arr2)