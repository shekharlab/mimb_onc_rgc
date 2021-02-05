import numpy as np
import pandas as pd
import seaborn as sns
import scipy.stats as sp
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import gridspec
import xgboost as xgb
from sklearn.metrics import confusion_matrix, adjusted_rand_score, accuracy_score
from sklearn.utils import sparsefuncs
import statsmodels.api as sm

import scanpy as sc
sc.settings.verbosity = 3
sc.settings.set_figure_params(dpi=80)

def xgbtrainatlas(
    train_anndata,
    test_anndata,
    genes,
    max_cells_per_ident = 1000,
    train_frac = 0.9
    ): 
    
    numbertrainclasses = len(train_anndata.obs.Type_num.values.categories.values)

    #Train XGBoost on 90% of training data and validate on the remaining data
    genes_index_train = []
    for i in genes:
        genes_index_train.append(np.where(train_anndata.var.index.values == i)[0][0])

    training_set_train_90 = []
    validation_set_train_10 = []
    training_label_train_90 = []
    validation_label_train_10 = []

    for i in train_anndata.obs.Type_num.values.categories.values:
        cells_in_clust = train_anndata.obs.index[train_anndata.obs.Type_num.values == i]
        n = min(max_cells_per_ident,round(len(cells_in_clust)*train_frac))
        train_temp = np.random.choice(cells_in_clust,n,replace = False)
        validation_temp = np.setdiff1d(cells_in_clust, train_temp)
        if len(train_temp) < 100:
            train_temp_bootstrap = np.random.choice(train_temp, size = 500 - int(len(train_temp)))
            train_temp = np.hstack([train_temp_bootstrap, train_temp])
        training_set_train_90 = np.hstack([training_set_train_90,train_temp])
        validation_set_train_10 = np.hstack([validation_set_train_10,validation_temp])
        training_label_train_90 = np.hstack([training_label_train_90,np.repeat(int(i[1:])-1,len(train_temp))])
        validation_label_train_10 = np.hstack([validation_label_train_10,np.repeat(int(i[1:])-1,len(validation_temp))])

    train_index_train_90 = []
    for i in training_set_train_90:
        train_index_train_90.append(np.where(train_anndata.obs.index.values == i)[0][0])
    validation_index_train_10 = []
    for i in validation_set_train_10:
        validation_index_train_10.append(np.where(train_anndata.obs.index.values == i)[0][0])

    xgb_params_train = {
        'objective':'multi:softprob',
        'eval_metric':'mlogloss',
        'num_class':numbertrainclasses,
        'eta':0.2,
        'max_depth':6,
        'subsample': 0.6}
    nround = 200

    freqs_dict = {x:list(training_label_train_90).count(x) for x in set(training_label_train_90)}
    freqs = pd.DataFrame(freqs_dict.items(), columns = ['Cluster', 'Freq'])
    minf = freqs.Freq.min()
    maxf = freqs.Freq.max()
    weights1 = []
    for i in training_label_train_90:
        f = freqs.Freq[i]
        weights1.append(0.6 + 0.4*(f-minf)/(maxf-minf))

    train_matrix_train_90 = xgb.DMatrix(data = train_anndata.X[:,genes_index_train][train_index_train_90,:], label = training_label_train_90, weight = weights1)
    validation_matrix_train_10 = xgb.DMatrix(data = train_anndata.X[:,genes_index_train][validation_index_train_10,:], label = validation_label_train_10)

    bst_model_train_90 = xgb.train(
        params = xgb_params_train,
        dtrain = train_matrix_train_90,
        num_boost_round = nround
        )

    validation_pred_train_10 = bst_model_train_90.predict(data = validation_matrix_train_10)

    valid_predlabels_train_10 = np.zeros((validation_pred_train_10.shape[0]))
    for i in range(validation_pred_train_10.shape[0]):
        valid_predlabels_train_10[i] = np.argmax(validation_pred_train_10[i,:])

    del train_matrix_train_90, validation_matrix_train_10, validation_pred_train_10

    #Predict the testing cluster labels
    genes_index_test = []
    for i in genes:
        genes_index_test.append(np.where(test_anndata.var.index.values == i)[0][0])

    full_testing_data = xgb.DMatrix(data = test_anndata.X[:,genes_index_test])
    test_prediction = bst_model_train_90.predict(data = full_testing_data)

    del bst_model_train_90, full_testing_data

    test_predlabels = np.zeros((test_prediction.shape[0]))
    for i in range(test_prediction.shape[0]):
        if np.max(test_prediction[i,:]) <= 40:
            if np.max(test_prediction[i,:]) > 0.7:
                test_predlabels[i] = np.argmax(test_prediction[i,:])
            else:
                test_predlabels[i] = numbertrainclasses
        else:
            if np.max(test_prediction[i,:]) > 0.5:
                test_predlabels[i] = np.argmax(test_prediction[i,:])
            else:
                test_predlabels[i] = numbertrainclasses

    return validation_label_train_10, valid_predlabels_train_10, test_predlabels

def xgbtrain(
    train_anndata,
    test_anndata,
    genes,
    max_cells_per_ident = 1000,
    train_frac = 0.9
    ): 
    
    numbertrainclasses = len(train_anndata.obs.Type_iGB.values.categories.values)

    #Train XGBoost on 90% of training data and validate on the remaining data
    genes_index_train = []
    for i in genes:
        genes_index_train.append(np.where(train_anndata.var.index.values == i)[0][0])

    training_set_train_90 = []
    validation_set_train_10 = []
    training_label_train_90 = []
    validation_label_train_10 = []

    for i in train_anndata.obs.Type_iGB.values.categories.values:
        cells_in_clust = train_anndata.obs.index[train_anndata.obs.Type_iGB.values == i]
        n = min(max_cells_per_ident,round(len(cells_in_clust)*train_frac))
        train_temp = np.random.choice(cells_in_clust,n,replace = False)
        validation_temp = np.setdiff1d(cells_in_clust, train_temp)
        if len(train_temp) < 100:
            train_temp_bootstrap = np.random.choice(train_temp, size = 500 - int(len(train_temp)))
            train_temp = np.hstack([train_temp_bootstrap, train_temp])
        training_set_train_90 = np.hstack([training_set_train_90,train_temp])
        validation_set_train_10 = np.hstack([validation_set_train_10,validation_temp])
        training_label_train_90 = np.hstack([training_label_train_90,np.repeat(int(i)-1,len(train_temp))])
        validation_label_train_10 = np.hstack([validation_label_train_10,np.repeat(int(i)-1,len(validation_temp))])

    train_index_train_90 = []
    for i in training_set_train_90:
        train_index_train_90.append(np.where(train_anndata.obs.index.values == i)[0][0])
    validation_index_train_10 = []
    for i in validation_set_train_10:
        validation_index_train_10.append(np.where(train_anndata.obs.index.values == i)[0][0])

    xgb_params_train = {
        'objective':'multi:softprob',
        'eval_metric':'mlogloss',
        'num_class':numbertrainclasses,
        'eta':0.2,
        'max_depth':6,
        'subsample': 0.6}
    nround = 200

    freqs_dict = {x:list(training_label_train_90).count(x) for x in set(training_label_train_90)}
    freqs = pd.DataFrame(freqs_dict.items(), columns = ['Cluster', 'Freq'])
    minf = freqs.Freq.min()
    maxf = freqs.Freq.max()
    weights1 = []
    for i in training_label_train_90:
        f = freqs.Freq[i]
        weights1.append(0.6 + 0.4*(f-minf)/(maxf-minf))

    train_matrix_train_90 = xgb.DMatrix(data = train_anndata.X[:,genes_index_train][train_index_train_90,:], label = training_label_train_90, weight = weights1)
    validation_matrix_train_10 = xgb.DMatrix(data = train_anndata.X[:,genes_index_train][validation_index_train_10,:], label = validation_label_train_10)

    bst_model_train_90 = xgb.train(
        params = xgb_params_train,
        dtrain = train_matrix_train_90,
        num_boost_round = nround
        )

    validation_pred_train_10 = bst_model_train_90.predict(data = validation_matrix_train_10)

    valid_predlabels_train_10 = np.zeros((validation_pred_train_10.shape[0]))
    for i in range(validation_pred_train_10.shape[0]):
        valid_predlabels_train_10[i] = np.argmax(validation_pred_train_10[i,:])

    del train_matrix_train_90, validation_matrix_train_10, validation_pred_train_10

    #Predict the testing cluster labels
    genes_index_test = []
    for i in genes:
        genes_index_test.append(np.where(test_anndata.var.index.values == i)[0][0])

    full_testing_data = xgb.DMatrix(data = test_anndata.X[:,genes_index_test])
    test_prediction = bst_model_train_90.predict(data = full_testing_data)

    del bst_model_train_90, full_testing_data

    test_predlabels = np.zeros((test_prediction.shape[0]))
    for i in range(test_prediction.shape[0]):
        if np.max(test_prediction[i,:]) <= 40:
            if np.max(test_prediction[i,:]) > 0.7:
                test_predlabels[i] = np.argmax(test_prediction[i,:])
            else:
                test_predlabels[i] = numbertrainclasses
        else:
            if np.max(test_prediction[i,:]) > 0.5:
                test_predlabels[i] = np.argmax(test_prediction[i,:])
            else:
                test_predlabels[i] = numbertrainclasses

    return validation_label_train_10, valid_predlabels_train_10, test_predlabels

def plotValidationConfusionMatrix(
    ytrue,
    ypred,
    save_as,
    title = '',
    xaxislabel = '',
    yaxislabel = ''
    ):

    confusionmatrix = confusion_matrix(y_true = ytrue, y_pred = ypred)
    confmatpercent = np.zeros(confusionmatrix.shape)
    for i in range(confusionmatrix.shape[0]):
        if np.sum(confusionmatrix[i,:]) != 0:
            confmatpercent[i,:] = confusionmatrix[i,:]/np.sum(confusionmatrix[i,:])
        else:
            confmatpercent[i,:] = confusionmatrix[i,:]
    diagcm = confmatpercent
    xticks = np.linspace(0, confmatpercent.shape[1]-1, confmatpercent.shape[1], dtype = int)
    dot_max = np.max(diagcm.flatten())
    dot_min = 0
    if dot_min != 0 or dot_max != 1:
        frac = np.clip(diagcm, dot_min, dot_max)
        old_range = dot_max - dot_min
        frac = (frac - dot_min) / old_range
    else:
        frac = diagcm
    xvalues = []
    yvalues = []
    sizes = []
    for i in range(diagcm.shape[0]):
        for j in range(diagcm.shape[1]):
            xvalues.append(j)
            yvalues.append(i)
            sizes.append((frac[i,j]*35)**1.5)
    size_legend_width = 0.5
    height = diagcm.shape[0] * 0.3 + 1
    height = max([1.5, height])
    heatmap_width = diagcm.shape[1] * 0.35
    width = (
        heatmap_width
        + size_legend_width
        )
    fig = plt.figure(figsize=(width, height))
    axs = gridspec.GridSpec(
        nrows=2,
        ncols=2,
        wspace=0.02,
        hspace=0.04,
        width_ratios=[
                    heatmap_width,
                    size_legend_width
                    ],
        height_ratios = [0.5, 10]
        )
    dot_ax = fig.add_subplot(axs[1, 0])
    dot_ax.scatter(xvalues,yvalues, s = sizes, c = 'blue', norm=None, edgecolor='none')
    y_ticks = range(diagcm.shape[0])
    dot_ax.set_yticks(y_ticks)
    dot_ax.set_yticklabels(y_ticks)
    x_ticks = range(diagcm.shape[1])
    dot_ax.set_xticks(x_ticks)
    dot_ax.set_xticklabels(xticks, rotation=90)
    dot_ax.tick_params(axis='both', labelsize='small')
    dot_ax.grid(True, linewidth = 0.2)
    dot_ax.set_axisbelow(True)
    dot_ax.set_xlim(-0.5, diagcm.shape[1] + 0.5)
    ymin, ymax = dot_ax.get_ylim()
    dot_ax.set_ylim(ymax + 0.5, ymin - 0.5)
    dot_ax.set_xlim(-1, diagcm.shape[1])
    dot_ax.set_xlabel(xaxislabel)
    dot_ax.set_ylabel(yaxislabel)
    dot_ax.set_title(title)
    size_legend_height = min(1.75, height)
    wspace = 10.5 / width
    axs3 = gridspec.GridSpecFromSubplotSpec(
        2,
        1,
        subplot_spec=axs[1, 1],
        wspace=wspace,
        height_ratios=[
                    size_legend_height / height,
                    (height - size_legend_height) / height
                    ]
        )
    diff = dot_max - dot_min
    if 0.3 < diff <= 0.6:
        step = 0.1
    elif diff <= 0.3:
        step = 0.05
    else:
        step = 0.2
    fracs_legends = np.arange(dot_max, dot_min, step * -1)[::-1]
    if dot_min != 0 or dot_max != 1:
        fracs_values = (fracs_legends - dot_min) / old_range
    else:
        fracs_values = fracs_legends
    size = (fracs_values * 35) ** 1.5
    size_legend = fig.add_subplot(axs3[0])
    size_legend.scatter(np.repeat(0, len(size)), range(len(size)), s=size, c = 'blue')
    size_legend.set_yticks(range(len(size)))
    labels = ["{:.0%}".format(x) for x in fracs_legends]
    if dot_max < 1:
        labels[-1] = ">" + labels[-1]
    size_legend.set_yticklabels(labels)
    size_legend.set_yticklabels(["{:.0%}".format(x) for x in fracs_legends])
    size_legend.tick_params(axis='y', left=False, labelleft=False, labelright=True)
    size_legend.tick_params(axis='x', bottom=False, labelbottom=False)
    size_legend.spines['right'].set_visible(False)
    size_legend.spines['top'].set_visible(False)
    size_legend.spines['left'].set_visible(False)
    size_legend.spines['bottom'].set_visible(False)
    size_legend.grid(False)
    ymin, ymax = size_legend.get_ylim()
    size_legend.set_ylim(ymin, ymax + 0.5)
    fig.savefig(save_as, bbox_inches = 'tight')

    return diagcm, xticks, axs