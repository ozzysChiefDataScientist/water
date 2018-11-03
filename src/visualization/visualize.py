import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.cm as cm
import matplotlib.pyplot as plt

def single_shap_plot(featureInd,shapDF,friendlyName):
    '''
    Recreates the visual created by shap's summary_plot(), but limited only to the feature noted in featureInd
    
    Parameters
    ----------
    featureInd : 0-index integer of the variable for plotting
    shapDF : A Pandas data frame produced by data.generate_shap_df()
    friendlyName : Friendly name of the variable for potting
    
    Returns
    ----------
    plt : A matplotlib object with a shap visualization of a single variable
    '''
    
    # Lines 26-39 from shap
    # Link: https://github.com/slundberg/shap/blob/010a607dbb919632773539523619a88c7cef4906/shap/plots/colors.py#L12-L25
    # Retrieval Date: 9/10/18
    
    red_blue = LinearSegmentedColormap('red_blue', { # #1E88E5 -> #ff0052
                                       'red': ((0.0, 30./255, 30./255),
                                               (1.0, 255./255, 255./255)),
                                       
                                       'green': ((0.0, 136./255, 136./255),
                                                 (1.0, 13./255, 13./255)),
                                       
                                       'blue': ((0.0, 229./255, 229./255),
                                                (1.0, 87./255, 87./255)),
                                       
                                       'alpha': ((0.0, 1, 1),
                                                 (0.5, 0.3, 0.3),
                                                 (1.0, 1, 1))
                                       })
                                       
    SMALL_SIZE = 14
    MEDIUM_SIZE = 16
    BIGGER_SIZE = 18

    alpha = 1
    plt.figure(figsize=(15,7))
    
    # Lines 37-44 adapted from shap
    # Link: https://github.com/slundberg/shap/blob/dda27ca1090167e4a2975a0ce75760fe6a45adec/shap/plots/summary.py#L188
    # Retrieval Date: 9/10/18
    plt.scatter(shapDF[shapDF['feature']==featureInd]['shap'],
                shapDF[shapDF['feature']==featureInd]['ys'],
                cmap=red_blue,
                vmin=min(shapDF[shapDF['feature']==featureInd]['vmin']),
                vmax=max(shapDF[shapDF['feature']==featureInd]['vmax']),
                c=shapDF[shapDF['feature']==featureInd]['values'],
                s=16, alpha=alpha, linewidth=0,
                zorder=3, rasterized=len(shapDF[shapDF['feature']==featureInd]['shap']) > 500)
    plt.xlabel("Influence on Model\nNegative values mean that the likelihood of an error DECREASES\nPositive values mean that the likelihood of an error INCREASES")
    plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels

    # Lines 55-64 from shap
    # Link: https://github.com/slundberg/shap/blob/dda27ca1090167e4a2975a0ce75760fe6a45adec/shap/plots/summary.py#L364
    # Retrieval Date: 9/10/18
    m = cm.ScalarMappable(cmap=red_blue)
    m.set_array([0, 1])
    cb = plt.colorbar(m, ticks=[0, 1], aspect=1000)
    cb.ax.tick_params(labelsize=15, length=0)
    cb.set_ticklabels(["Low value", "High value"])
    cb.set_alpha(1)
    cb.set_label("{}".format(friendlyName), size=12, labelpad=0)
    cb.outline.set_visible(False)
    bbox = cb.ax.get_window_extent().transformed(plt.gcf().dpi_scale_trans.inverted())
    cb.ax.set_aspect((bbox.height - 0.9) * 20)
    return plt
