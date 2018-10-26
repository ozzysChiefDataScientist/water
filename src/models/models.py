import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_curve

def gen_train_test_ids(df,idCol,seed=122,trainPercent=.8):
    '''
    Randomly assigns PWSIDs to train/test.
        
    Parameters
    ----------
    df : A Pandas data frame
    idCol : String with name of column that uniquely identifies each water district
    seed : Integer, for reproducibility
    trainPercent : The percent of IDs in df to include in the training set
    
    Returns
    ----------
    train_ids : PWSIDs of training set
    test_ids : PWSIDs of test set
    '''
    
    ids = set(df[idCol]).intersection(set(df[idCol]))
    ids = sorted(list(ids))
    
    np.random.seed(seed)
    train_indices =  np.random.choice(np.arange(0,len(ids)),size=int(trainPercent*len(ids)),replace=False)
    train_ids = list(np.array(ids)[[train_indices]])
    
    test_ids = list(set(ids).difference(set(train_ids)))
    
    return train_ids, test_ids

def generate_cv_folds(trainDF,trainIDs,trainIDCol,seed=122):
    '''
    Parameters
    ----------
    trainDF : A Pandas data frame
    trainIDs : A list with strings identifying rows in the train set
    trainIDCol : A string with the name of the column identifying rows in training set
    seed : Integer, for reproducibility
    
    Returns
    ----------
    trainDF : trainDF with a column random_group. PWSIDs are randomly assigned into 5 random_group buckets; these buckets are used as cross-validation folds.
    '''
    trainGroup = pd.DataFrame({trainIDCol:trainIDs})
    np.random.seed(seed)
    trainGroup['random'] = np.random.uniform(size=trainGroup.shape[0])
    trainGroup['random_group'] = 0
    trainGroup.loc[trainGroup['random'] < .2, 'random_group'] = 1
    trainGroup.loc[(trainGroup['random'] < .4) & (trainGroup['random'] >= .2), 'random_group'] = 2
    trainGroup.loc[(trainGroup['random'] < .6) & (trainGroup['random'] >= .4), 'random_group'] = 3
    trainGroup.loc[(trainGroup['random'] < .8) & (trainGroup['random'] >= .6), 'random_group'] = 4
    trainDF = trainDF.merge(trainGroup[['PWSID','random_group']],on=trainIDCol)
    return trainDF

def generate_train_test_df_and_matrix(df,idCol,trainIDs,testIDs,predictors,responseCol):
    '''
    Splits the data set into train and test; returns as both matrix ready for XGBoost and as Pandas data frame
        
    Parameters
    ----------
    df : A Pandas data frame
    idCol : A string with the name of the column identifying rows in training set
    trainIDs : A list with strings identifying rows in the train set
    testIDs : A list with strings identifying rows in the test set
    predictors : A string with the name of the column identifying rows in training set
    responseCol : A string with the name of the response variable column
    
    Returns
    ----------
    train_X : A Pandas data frame of observations in the training set
    test_X : A Pandas data frame of observations in the test set
    xgb_train_X : A matrix version of train_X, limited to predictive variable columns
    xgb_test_X : A matrix version of test_X, limited to predictive variable columns
    xgb_train_y : A Pandas series of the response in train_X
    xgb_test_y : A Pandas series of the response in test_X
    '''
    train_X = df[df[idCol].isin(trainIDs)]
    test_X = df[df[idCol].isin(testIDs)]
    
    xgb_train_X = train_X[predictors].as_matrix()
    xgb_test_X = test_X[predictors].as_matrix()

    xgb_train_y = train_X[responseCol]
    xgb_test_y = test_X[responseCol]

    return train_X, test_X, xgb_train_X, xgb_test_X, xgb_train_y, xgb_test_y

def review_train_test_split(train,test,responseCol):
    '''
    Describes the train and test sets
        
    Parameters
    ----------
    train : A Pandas data frame
    test : A list with strings identifying rows in the train set
    responseCol : A string with the name of the response variable column
    '''
    print("Train shape: {}".format(train.shape))
    print("Test shape: {}".format(test.shape))

    # What % of records have units reported incorrectly?
    print("% incorrect in train: {}".format(train[responseCol].value_counts() / train.shape[0]))

    # What % of records have units reported incorrectly?
    print("% incorrect in test: {}".format(test[responseCol].value_counts() / test.shape[0]))

def gridsearch_best_precision_and_recall(gbm,parameters,gss,trainDF,wholeDF,trainIDs,responseCol,xgb_train_X,xgb_test_X,xgb_train_y,xgb_test_y):
    '''
    Performs grid search to test for best parameters. Tests for precision, then for recall.
        
    Parameters
    ----------
    gbm : XGBClassifier object
    parameters : a dictionary specifying hyperparameters to tune
    gss : A GroupShuffleSplit object
    trainDF : A Pandas data frame with the training set
    wholeDF : A Pandas data frame with the train and test sets
    trainIDs : A list with strings identifying rows in the train set
    responseCol : A string with the name of the response variable column
    xgb_train_X : A matrix version of trainDF, limited to predictive variable columns
    xgb_test_X : A matrix version of the test set, limited to predictive variable columns
    xgb_train_y : A Pandas series of the response in trainDF
    xgb_test_y : A Pandas series of the response in the test set
    '''
    
    # Lines 142-143 adapted from sklearn
    # Lines 136-141, Lines 144-162 from sklearn
    # Link: https://github.com/scikit-learn/scikit-learn/blob/master/examples/model_selection/plot_grid_search_digits.py#L47-L77
    # Retrieval Date: 9/25/18
    scores = ['precision', 'recall']

    grid_search_results = []

    for score in scores:
        print("# Tuning hyper-parameters for %s" % score)
        print()
    
        clf = GridSearchCV(gbm, parameters, n_jobs=5, scoring='%s_weighted' % score,verbose=2,refit=True,iid=False,cv=list(gss.split(trainDF, wholeDF[wholeDF['PWSID'].isin(trainIDs)][responseCol],groups=trainDF['random_group'])) )
        clf.fit(xgb_train_X, xgb_train_y)
                                           
        print("Best parameters set found on development set:")
        print()
        print(clf.best_params_)
        print()
        print("Grid scores on development set:")
        print()
        for params, mean_score, scores in clf.grid_scores_:
            print("%0.3f (+/-%0.03f) for %r" % (mean_score, scores.std() / 2, params))
            print()

        print("Detailed classification report:")
        print()
        print("The model is trained on the full development set.")
        print("The scores are computed on the full evaluation set.")
        print()
        y_true, y_pred = xgb_test_y, clf.predict(xgb_test_X)
        print(classification_report(y_true, y_pred))
        print()

        grid_search_results = grid_search_results + [clf,]
    return grid_search_results

def get_precision_recall(gbm,y,X,beta=1):
    '''
    Assesses precision, recall, and f1 score at various thresholds.
    
    Parameters
    ----------
    gbm : A fitted XGBoost model
    y : A Pandas series of the response
    X : A matrix of the predictor variables
    beta : Beta determines the weight of precision in the f1 score. Beta > 1 gives more weight to recall. Precision and recall are weighted equally when beta=1.
    
    Returns
    ----------
    precisionRecallDF : A data frame summarizing precision, recall,  f1 score with beta=1, and f1 score at specified beta at various thresholds.
    
    '''
    precision, recall, thresholds = precision_recall_curve( y,  list(pd.DataFrame(gbm.predict_proba(X))[1]))
    thresholds = [0,] + list(thresholds)
    precisionRecallDF = pd.DataFrame({"precision":precision,"recall":recall,"threshold":thresholds})
    precisionRecallDF['fl'] = 2*((precisionRecallDF['precision']*precisionRecallDF['recall'])/(precisionRecallDF['precision']+precisionRecallDF['recall']))
    precisionRecallDF['fl_beta'] = (1+beta**2)* ((precisionRecallDF['precision']*precisionRecallDF['recall'])/((precisionRecallDF['precision'] * beta**2) +precisionRecallDF['recall'] ))
    return precisionRecallDF
