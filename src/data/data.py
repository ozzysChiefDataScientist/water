import pandas as pd

def add_one_hot_encoding(series,df,predictors):
    '''
    Parameters
    ----------
    series : A Pandas series with the variable to be one-hot encoded
    df : A Pandas data frame
    predictors : A list with names of predictor variables
    
    Returns
    ----------
    df: A Pandas data frame with the series variable represented in one-hot encoding
    '''
    one_hot = pd.get_dummies(series)
    predictors = predictors +  list(one_hot.columns.values)
    df = df.join(one_hot)
    return df,predictors

def compare_correct_incorrect_values(df,predictor):
    '''
    Parameters
    ----------
    df : A Pandas data frame
    predictors : A string with name of predictor variable column
    '''
    incorrectValues = df[(pd.isnull(df[predictor])==False)
                  & (df['original_units_incorrect']==1)][predictor].describe(percentiles=[0.05,0.1,0.5,0.9,0.95]).reset_index()
    correctValues = df[(pd.isnull(df[predictor])==False)
                  & (df['original_units_incorrect']==0)][predictor].describe(percentiles=[0.05,0.1,0.5,0.9,0.95]).reset_index()
    compare = incorrectValues.merge(correctValues,on="index",suffixes=['_incorrect','_correct'])
    return compare

