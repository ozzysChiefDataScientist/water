import numpy as np
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
    
    Returns
    ----------
    df: A data frame with the following columns: index (notes the measurement); gpcd_from_water_deliv_residential_revised_units_in_orig_units_in_gallons_incorrect (value of metric specified in index on incorrect reports); gpcd_from_water_deliv_residential_revised_units_in_orig_units_in_gallons_correct (value of metric specified in index on correct reports)
    '''
    incorrectValues = df[(pd.isnull(df[predictor])==False)
                  & (df['original_units_incorrect']==1)][predictor].describe(percentiles=[0.05,0.1,0.5,0.9,0.95]).reset_index()
    correctValues = df[(pd.isnull(df[predictor])==False)
                  & (df['original_units_incorrect']==0)][predictor].describe(percentiles=[0.05,0.1,0.5,0.9,0.95]).reset_index()
    compare = incorrectValues.merge(correctValues,on="index",suffixes=['_incorrect','_correct'])
    return compare

def generate_shap_df(shap_values,xgb_train_X):
    '''
    The shap.summary_plot() function shows all variables in a model. This function returns the raw values displayed in the shap.summary_plot() plot for users who need to explore a specific variable more closely.
        
    Parameters
    ----------
    shap_values : Shap values created from shap.TreeExplainer.shap_values(X) on a dataset X
    xgb_train_X : The training data set, in matrix form
    
    Returns
    ----------
    shapDF: A dataframe with the values plotted in shap.summary_plot()
    '''
    shapDF = pd.DataFrame()
    max_display = 7
    row_height = 0.4
    feature_order = np.argsort(np.sum(np.abs(shap_values), axis=0))
    feature_order = feature_order[-min(max_display, len(feature_order)):]
    features = xgb_train_X
    
    # Lines 60-100 from https://github.com/slundberg/shap/blob/master/shap/plots/summary.py#L187-L248
    for pos, i in enumerate(feature_order):
        shaps = shap_values[:, i]
        values = None if features is None else features[:, i]
        inds = np.arange(len(shaps))
        np.random.shuffle(inds)
        if values is not None:
            values = values[inds]
        shaps = shaps[inds]
        colored_feature = True
        try:
            values = np.array(values, dtype=np.float64)  # make sure this can be numeric
        except:
            colored_feature = False
        N = len(shaps)
        nbins = 100
        quant = np.round(nbins * (shaps - np.min(shaps)) / (np.max(shaps) - np.min(shaps) + 1e-8))
        inds = np.argsort(quant + np.random.randn(N) * 1e-6)
        layer = 0
        last_bin = -1
        ys = np.zeros(N)
        for ind in inds:
            if quant[ind] != last_bin:
                layer = 0
            ys[ind] = np.ceil(layer / 2) * ((layer % 2) * 2 - 1)
            layer += 1
            last_bin = quant[ind]
        ys *= 0.9 * (row_height / np.max(ys + 1))
        
        if features is not None and colored_feature:
            # trim the color range, but prevent the color range from collapsing
            vmin = np.nanpercentile(values, 5)
            vmax = np.nanpercentile(values, 95)
            if vmin == vmax:
                vmin = np.nanpercentile(values, 1)
                vmax = np.nanpercentile(values, 99)
                if vmin == vmax:
                    vmin = np.min(values)
                    vmax = np.max(values)
        
            assert features.shape[0] == len(shaps), "Feature and SHAP matrices must have the same number of rows!"
        nan_mask = np.isnan(values)
        
        if shapDF.shape[0]==0:
            shapDF = pd.DataFrame({"shap":shaps[np.invert(nan_mask)],
                                  "ys":ys[np.invert(nan_mask)],
                                  "feature":i,
                                  "vmin": vmin,
                                  "vmax": vmax,
                                  "values":  values[np.invert(nan_mask)]
                                  })
        else:
            temp = pd.DataFrame({"shap":shaps[np.invert(nan_mask)],
                                "ys":ys[np.invert(nan_mask)],
                                "feature":i,
                                "vmin": vmin,
                                "vmax": vmax,
                                "values":  values[np.invert(nan_mask)]
                                })
            shapDF = shapDF.append(temp)
    return shapDF
