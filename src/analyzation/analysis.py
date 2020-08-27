import pandas as pd
import numpy as np

lending_data_info = pd.read_csv('../data/raw/lending_club_info.csv', index_col='LoanStatNew')

def feat_info(col_name):
    '''
    Takes the column name or list of names
    and prints their features
    '''
    print(lending_data_info.loc[col_name]['Description'])
    
def filler (total_acc, mort_acc, replace):
    '''
    replaces the missing values in mort_acc with median
    of total_acc.
    '''
    if np.isnan(mort_acc):
        return replace
    else:
        return mort_acc

def drop_columns(data, cols):
    '''
    drops the specified columns
    '''
    data = data.drop(cols, axis=1)
    return data