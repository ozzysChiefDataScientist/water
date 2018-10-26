import pandas as pd

def test_verify_unique_rows(df):
    '''
    Confirm that the data set has one row per water system per date
    '''
    df['key'] = df[['Water.System.Name','Date']].apply(lambda x: "{}_{}".format(x[0],x[1]),axis=1)
    keyCounts = df['key'].value_counts().reset_index()
    assert keyCounts[keyCounts['key']>1].shape[0]==0

def verify_train_test_split(train_ids,test_ids):
    '''
    Confirms that PWSIDs are distinct between the training and test set
    
    Parameters
    ----------
    train_ids : List or series of PWSID in training set
    test_ids : List or series of PWSID in training set
    '''
    assert len(set(train_ids).intersection(set(test_ids)))==0
