import pandas as pd

import sys

sys.path.insert(1, '../src')
from analyzation import analysis as al

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

def pre_processing(data):
    #Lets convert months to an int value
    data['term'] = data['term'].apply(lambda x: int(x.split()[0]))
    
    #Lets try to grab the year form this data
    data['earliest_cr_line'] = data['earliest_cr_line'].apply(lambda x: int(x[-4:]))

    #Lets grab the zipcode from the address
    data['address'] = data['address'].apply(lambda x: int(x[-5:]))
    
    #Lets just replace "NONE" and "ANY" with another
    data["home_ownership"] = data["home_ownership"].replace(["NONE", "ANY"], "OTHER")
    
    #Getting dummies
    dummies = pd.get_dummies(data[['verification_status',
                               'initial_list_status', 
                                'purpose','application_type',
                                'home_ownership',
                                'sub_grade']], drop_first=True)
    data = pd.concat([data, dummies], axis=1)
    
    dummies_zipcode = pd.get_dummies(data['address'], drop_first=True)
    data = pd.concat([data, dummies_zipcode], axis=1)
    
    #Grade is part of subgrade so we can get rid of that
    #The new customers are not going to have issue_id
    
    data= data.drop(['issue_d', 'grade',
                                       'verification_status',
                                       'initial_list_status', 
                                        'purpose','application_type',
                                        'home_ownership',
                                        'address','sub_grade',
                                       'loan_status'], axis=1)
    return data

def pre_processing_split(data=(), scaler = MinMaxScaler(),split_size=(80,10,10)):
    '''
    Given the data this function splits the 
    data into split size. and uses min max scaler 
    to transform the data.
    '''
    test_size = split_size[2]/100
    val_size = split_size[1] / (split_size[0] + split_size[1])
    
    lend_feat_train, lend_feat_test, lend_paid_train, lend_paid_test = train_test_split(data[0],                                                                                           data[1], 
                                                    test_size=test_size, random_state=42)

    lend_feat_train, lend_feat_valid, lend_paid_train, lend_paid_valid = train_test_split(data[0],                                                                                           data[1], 
                                                  test_size=val_size, random_state=42)
    
    lend_feat_train = scaler.fit_transform(lend_feat_train)
    lend_feat_valid = scaler.transform(lend_feat_valid)
    lend_feat_test = scaler.transform(lend_feat_test)
    
    return lend_feat_train, lend_feat_valid, lend_feat_test, lend_paid_train, lend_paid_valid, lend_paid_test
    