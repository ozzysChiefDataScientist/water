import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.cm as cm
import matplotlib.pyplot as plt

def single_shap_plot(featureInd,shapDF,friendlyName):
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

    m = cm.ScalarMappable(cmap=red_blue)
    m.set_array([0, 1])
    cb = plt.colorbar(m, ticks=[0, 1], aspect=1000)
    cb.ax.tick_params(labelsize=15, length=0)
    cb.set_ticklabels(["Low value", "High value"])
    
    cb.set_label("{}".format(friendlyName), size=12, labelpad=0)
    cb.set_alpha(1)
    cb.outline.set_visible(False)
    bbox = cb.ax.get_window_extent().transformed(plt.gcf().dpi_scale_trans.inverted())
    cb.ax.set_aspect((bbox.height - 0.9) * 20)
    return plt

def shap_dependence_plot(ind, shap_values, features, feature_names=None, display_features=None,
                    interaction_index="auto", color="#1E88E5", axis_color="#333333",
                    dot_size=16, alpha=1, title=None, show=True):
    """
        Create a SHAP dependence plot, colored by an interaction feature.
        Parameters
        ----------
        ind : int
        Index of the feature to plot.
        shap_values : numpy.array
        Matrix of SHAP values (# samples x # features)
        features : numpy.array or pandas.DataFrame
        Matrix of feature values (# samples x # features)
        feature_names : list
        Names of the features (length # features)
        display_features : numpy.array or pandas.DataFrame
        Matrix of feature values for visual display (such as strings instead of coded values)
        interaction_index : "auto", None, or int
        The index of the feature used to color the plot.
    """
    
    labels = {
        'MAIN_EFFECT': "SHAP main effect value for\n%s",
        'INTERACTION_VALUE': "SHAP interaction value",
        'INTERACTION_EFFECT': "SHAP interaction value for\n%s and %s",
        'VALUE': "SHAP value (impact on model output)",
        'GLOBAL_VALUE': "mean(|SHAP value|) (average impact on model output magnitude)",
        'VALUE_FOR': "SHAP value for\n%s",
        'PLOT_FOR': "SHAP plot for %s",
        'FEATURE': "Feature %s",
        'FEATURE_VALUE': "Feature value",
        'FEATURE_VALUE_LOW': "Low",
        'FEATURE_VALUE_HIGH': "High",
        'JOINT_VALUE': "Joint SHAP value"
    }
    
    # convert from DataFrames if we got any
    if str(type(features)).endswith("'pandas.core.frame.DataFrame'>"):
        if feature_names is None:
            feature_names = features.columns
        features = features.values
    if str(type(display_features)).endswith("'pandas.core.frame.DataFrame'>"):
        if feature_names is None:
            feature_names = display_features.columns
        display_features = display_features.values
    elif display_features is None:
        display_features = features
    
    if feature_names is None:
        feature_names = [labels['FEATURE'] % str(i) for i in range(shap_values.shape[1])]
    
    # allow vectors to be passed
    if len(shap_values.shape) == 1:
        shap_values = np.reshape(shap_values, len(shap_values), 1)
    if len(features.shape) == 1:
        features = np.reshape(features, len(features), 1)

    def convert_name(ind):
        if type(ind) == str:
            nzinds = np.where(feature_names == ind)[0]
            if len(nzinds) == 0:
                print("Could not find feature named: " + ind)
                return None
            else:
                return nzinds[0]
        else:
            return ind

    ind = convert_name(ind)

    # plotting SHAP interaction values
    if len(shap_values.shape) == 3 and len(ind) == 2:
        ind1 = convert_name(ind[0])
        ind2 = convert_name(ind[1])
        if ind1 == ind2:
            proj_shap_values = shap_values[:, ind2, :]
        else:
            proj_shap_values = shap_values[:, ind2, :] * 2  # off-diag values are split in half

        # TODO: remove recursion; generally the functions should be shorter for more maintainable code
        dependence_plot(
                        ind1, proj_shap_values, features, feature_names=feature_names,
                        interaction_index=ind2, display_features=display_features, show=False
                        )
        if ind1 == ind2:
            pl.ylabel(labels['MAIN_EFFECT'] % feature_names[ind1])
        else:
            pl.ylabel(labels['INTERACTION_EFFECT'] % (feature_names[ind1], feature_names[ind2]))
        
        if show:
            pl.show()
        return

    assert shap_values.shape[0] == features.shape[0], \
        "'shap_values' and 'features' values must have the same number of rows!"
    assert shap_values.shape[1] == features.shape[1], \
        "'shap_values' must have the same number of columns as 'features'!"

    # get both the raw and display feature values
    xv = features[:, ind]
    xd = display_features[:, ind]
    s = shap_values[:, ind]
    if type(xd[0]) == str:
        name_map = {}
        for i in range(len(xv)):
            name_map[xd[i]] = xv[i]
        xnames = list(name_map.keys())

    # allow a single feature name to be passed alone
    if type(feature_names) == str:
        feature_names = [feature_names]
        name = feature_names[ind]
        
        # guess what other feature as the stongest interaction with the plotted feature
        if interaction_index == "auto":
            interaction_index = approx_interactions(ind, shap_values, features)[0]
    interaction_index = convert_name(interaction_index)
    categorical_interaction = False
        
    # get both the raw and display color values
    if interaction_index is not None:
        cv = features[:, interaction_index]
        cd = display_features[:, interaction_index]
        clow = np.nanpercentile(features[:, interaction_index].astype(np.float), 5)
        chigh = np.nanpercentile(features[:, interaction_index].astype(np.float), 95)
        if type(cd[0]) == str:
            cname_map = {}
            for i in range(len(cv)):
                cname_map[cd[i]] = cv[i]
            cnames = list(cname_map.keys())
            categorical_interaction = True
        elif clow % 1 == 0 and chigh % 1 == 0 and len(set(features[:, interaction_index])) < 50:
            categorical_interaction = True

    # discritize colors for categorical features
    color_norm = None
    if categorical_interaction and clow != chigh:
        bounds = np.linspace(clow, chigh, chigh - clow + 2)
        color_norm = matplotlib.colors.BoundaryNorm(bounds, colors.red_blue.N)

    # the actual scatter plot, TODO: adapt the dot_size to the number of data points?
    if interaction_index is not None:
        pl.scatter(xv, s, s=dot_size, linewidth=0, c=features[:, interaction_index], cmap=colors.red_blue,
                   alpha=alpha, vmin=clow, vmax=chigh, norm=color_norm, rasterized=len(xv) > 500)
    else:
        pl.scatter(xv, s, s=dot_size, linewidth=0, color="#1E88E5",alpha=alpha, rasterized=len(xv) > 500)

    if interaction_index != ind and interaction_index is not None:
        # draw the color bar
        if type(cd[0]) == str:
            tick_positions = [cname_map[n] for n in cnames]
            if len(tick_positions) == 2:
                tick_positions[0] -= 0.25
                tick_positions[1] += 0.25
                cb = pl.colorbar(ticks=tick_positions)
                cb.set_ticklabels(cnames)
            else:
                cb = pl.colorbar()
            
            cb.set_label(feature_names[interaction_index], size=13)
            cb.ax.tick_params(labelsize=11)
            if categorical_interaction:
                cb.ax.tick_params(length=0)
            cb.set_alpha(1)
            cb.outline.set_visible(False)
            bbox = cb.ax.get_window_extent().transformed(pl.gcf().dpi_scale_trans.inverted())
            cb.ax.set_aspect((bbox.height - 0.7) * 20)
        
        # make the plot more readable
        if interaction_index != ind:
            pl.gcf().set_size_inches(7.5, 5)
    else:
        pl.gcf().set_size_inches(6, 5)
        pl.xlabel(name, color=axis_color, fontsize=13)
        pl.ylabel(labels['VALUE_FOR'] % name, color=axis_color, fontsize=13)
        if title is not None:
            pl.title(title, color=axis_color, fontsize=13)
    pl.gca().xaxis.set_ticks_position('bottom')
    pl.gca().yaxis.set_ticks_position('left')
    pl.gca().spines['right'].set_visible(False)
    pl.gca().spines['top'].set_visible(False)
    pl.gca().tick_params(color=axis_color, labelcolor=axis_color, labelsize=11)
    for spine in pl.gca().spines.values():
        spine.set_edgecolor(axis_color)
        if type(xd[0]) == str:
            pl.xticks([name_map[n] for n in xnames], xnames, rotation='vertical', fontsize=11)

    return pl
