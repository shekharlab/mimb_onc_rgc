import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import scanpy as sc
import pandas as pd
from harmony import harmonize
from anndata import AnnData
import anndata
import seaborn as sns
from sklearn.utils import shuffle, sparsefuncs
import random

import matplotlib as mpl
from matplotlib import gridspec
import xgboost as xgb
from sklearn.metrics import confusion_matrix

from typing import Union, Optional, Tuple, Collection, Sequence, Iterable
from scipy.sparse import issparse, isspmatrix_csr, csr_matrix, spmatrix

sc.settings.verbosity = 3            # verbosity: errors (0), warnings (1), info (2), hints (3)
sc.set_figure_params(dpi=100, dpi_save=200)


#functions for generating diagonal dot plots
def diag_dotplot(adata):
    x = adata.raw.X.A
    comb = np.zeros((adata.shape[0], 45))
    
    #gene combination matrix
    comb[:,0] = x[:, adata.var_names.get_loc('Gpr88')]
    comb[two_POS(adata, 'Serpinb1b+Gm17750')[0],1] = two_POS(adata, 'Serpinb1b+Gm17750')[1],
    comb[:,2] = x[:, adata.var_names.get_loc('Mmp17')]
    comb[two_POS(adata, 'Lypd1+Ntrk1')[0],3] = two_POS(adata, 'Lypd1+Ntrk1')[1]
    comb[two_POS(adata, 'Cartpt+Vit')[0],4] = two_POS(adata, 'Cartpt+Vit')[1]
    comb[:,5] = x[:, adata.var_names.get_loc('Apela')]
    comb[two_POS(adata, 'Cartpt+Col25a1')[0],6] = two_POS(adata, 'Cartpt+Col25a1')[1]
    comb[two_POS(adata, 'Tbr1+Irx4')[0],7] = two_POS(adata, 'Tbr1+Irx4')[1]
    comb[two_POS(adata, 'Pcdh20+4833423E24Rik')[0],8] = two_POS(adata, 'Pcdh20+4833423E24Rik')[1]
    comb[three_pos(adata, 'Penk+Prdm8+Slc24a2')[0],9] = three_pos(adata, 'Penk+Prdm8+Slc24a2')[1]
    comb[two_POS(adata, 'Serpine2+Amigo2')[0],10] = two_POS(adata, 'Serpine2+Amigo2')[1]
    comb[two_POS(adata, 'Penk+Gal')[0],11] = two_POS(adata, 'Penk+Gal')[1]
    comb[two_POS(adata, 'Tbr1+Calca')[0],12] = two_POS(adata, 'Tbr1+Calca')[1]
    comb[two_POS(adata, 'Serpine2+Cdhr1')[0],13] = two_POS(adata, 'Serpine2+Cdhr1')[1]
    comb[:,14] = x[:, adata.var_names.get_loc('Prokr1')]
    comb[:,15] = x[:, adata.var_names.get_loc('Fam19a4')]
    comb[:,16] = x[:, adata.var_names.get_loc('Slc17a7')]
    comb[two_POS(adata, 'Penk+Igfbp5')[0],17] = two_POS(adata, 'Penk+Igfbp5')[1]
    comb[:,18] = x[:, adata.var_names.get_loc('Prkcg')]
    comb[two_POS(adata, 'Foxp2+Cdk15')[0],19] = two_POS(adata, 'Foxp2+Cdk15')[1]
    comb[two_POS(adata, 'Stxbp6+Prlr')[0],20] = two_POS(adata, 'Stxbp6+Prlr')[1]
    comb[pos_neg(adata, 'Lypd1-Ntrk1')[0],21] = pos_neg(adata, 'Lypd1-Ntrk1')[1]
    comb[:,22] = x[:, adata.var_names.get_loc('Postn')]
    comb[two_POS(adata, 'Tbx20+Spp1')[0],23] = two_POS(adata, 'Tbx20+Spp1')[1]
    comb[:,24] = x[:, adata.var_names.get_loc('Rhox5')]
    comb[pos_pos_neg(adata, ['Adcyap1', 'Opn4', 'Nmb'])[0],25] = pos_pos_neg(adata, ['Adcyap1', 'Opn4', 'Nmb'])[1]
    comb[pos_neg(adata, 'Tpbg-Spp1')[0],26] = pos_neg(adata, 'Tpbg-Spp1')[1]
    comb[two_POS(adata, 'Igfbp4+Chrm2')[0],27] = two_POS(adata, 'Igfbp4+Chrm2')[1]
    comb[two_POS(adata, 'Stxbp6+Coch')[0],28] = two_POS(adata, 'Stxbp6+Coch')[1]
    comb[:,29] = x[:, adata.var_names.get_loc('Ceacam10')]
    comb[two_POS(adata, 'Foxp2+Anxa3')[0],30] = two_POS(adata, 'Foxp2+Anxa3')[1]
    comb[two_POS(adata, 'Neurod2+S100b')[0],31] = two_POS(adata, 'Neurod2+S100b')[1]
    comb[two_POS(adata, 'Foxp2+Irx4')[0],32] = two_POS(adata, 'Foxp2+Irx4')[1]
    comb[:,33] = x[:, adata.var_names.get_loc('Nmb')]
    comb[two_POS(adata, 'Spp1+Kit')[0],34] = two_POS(adata, 'Spp1+Kit')[1]
    comb[two_POS(adata, 'Spp1+Fes')[0],35] = two_POS(adata, 'Spp1+Fes')[1]
    comb[two_POS(adata, 'Spp1+Il1rapl2')[0],36] = two_POS(adata, 'Spp1+Il1rapl2')[1]
    comb[two_POS(adata, 'Bhlhe22+Fxyd6')[0],37] = two_POS(adata, 'Bhlhe22+Fxyd6')[1]
    comb[two_POS(adata, 'Spp1+Tpbg')[0],38] = two_POS(adata, 'Spp1+Tpbg')[1]
    comb[:,39] = x[:, adata.var_names.get_loc('Pde1a')]
    comb[two_POS(adata, 'Tbr1+Pcdh20')[0],40] = two_POS(adata, 'Tbr1+Pcdh20')[1]
    comb[:,41] = x[:, adata.var_names.get_loc('Zic1')]
    comb[two_POS(adata, 'Tbx20+Tagln2')[0],42] = two_POS(adata, 'Tbx20+Tagln2')[1]
    
    comb[pos_pos_neg(adata, ['Prkcq', 'Tac1', 'Spp1'])[0],43] = pos_pos_neg(adata, ['Prkcq', 'Tac1', 'Spp1'])[1]
    
    comb[two_POS(adata, 'Slc7a11+Plpp4')[0],44] = two_POS(adata, 'Slc7a11+Plpp4')[1]


    comb_vars = ['Gpr88', 'Serpinb1b+Gm17750+', 'Mmp17', 'Lypd1+Ntrk1+', 'Cartpt+Vit+', 'Apela',
                'Cartpt+Col25a1+', 'Tbr1+Irx4+', 'Pcdh20+4833423E24Rik+', 'Penk+Prdm8+Slc24a2+',
                'Serpine2+Amigo2+', 'Penk+Gal+', 'Tbr1+Calca+', 'Serpine2+Cdhr1+', 'Prokr1', 'Fam19a4',
                'Slc17a7', 'Penk+Igfbp5+', 'Prkcg', 'Foxp2+Cdk15+', 'Stxbp6+Prlr+', 'Lypd1+Ntrk1-', 'Postn', 
                'Tbx20+Spp1+', 'Rhox5', 'Adcyap1+Opn4+Nmb-', 'Tpbg+Spp1-', 'Igfbp4+Chrm2+', 'Stxbp6+Coch+',
                'Ceacam10', 'Foxp2+Anxa3+', 'Neurod2+S100b+', 'Foxp2+Irx4+', 'Nmb', 'Spp1+Kit+', 'Spp1+Fes+',
                'Spp1+Il1rapl2+', 'Bhlhe22+Fxyd6+', 'Spp1+Tpbg+', 'Pde1a', 'Tbr1+Pcdh20+', 'Zic1', 'Tbx20+Tagln2+',
                'Prkcq+Tac1+Spp1-', 'Slc7a11+Plpp4+']

    adata_comb = anndata.AnnData(comb, obs=adata.obs)
    adata_comb.var_names = comb_vars
    
    return adata_comb

def two_POS(adata, gene_comb):
    counts = adata.raw.X.A
    split = gene_comb.split('+')
    
    inters_idxs = np.array(list(set(np.where(counts[:,adata.var.index.get_loc(split[0])]>0)[0]).intersection(np.where(counts[:,adata.var.index.get_loc(split[1])]>0)[0])))
    
    
    a = counts[inters_idxs.astype(int), adata.var.index.get_loc(split[0])]
    b = counts[inters_idxs.astype(int), adata.var.index.get_loc(split[1])]
    
    return [inters_idxs.astype(int), np.mean(np.array([a,b]), axis=0)]

def pos_neg(adata, gene_comb):
    counts = adata.raw.X.A
    split = gene_comb.split('-')
    
    inters_idxs = np.array(list(set(np.where(counts[:,adata.var.index.get_loc(split[0])]>0)[0]).intersection(np.where(counts[:,adata.var.index.get_loc(split[1])]==0)[0])))
    
    
    a = counts[inters_idxs.astype(int), adata.var.index.get_loc(split[0])]
    b = counts[inters_idxs.astype(int), adata.var.index.get_loc(split[1])]
    
    return [inters_idxs.astype(int), np.mean(np.array([a,b]), axis=0)]

def three_pos(adata, gene_comb):
    counts = adata.raw.X.A
    split = gene_comb.split('+')
    
    inters_idxs = np.array(list(set(np.where(counts[:,adata.var.index.get_loc(split[0])]>0)[0]) & set(np.where(counts[:,adata.var.index.get_loc(split[1])]>0)[0]) & set(np.where(counts[:,adata.var.index.get_loc(split[2])]>0)[0])))    
    
    a = counts[inters_idxs.astype(int), adata.var.index.get_loc(split[0])]
    b = counts[inters_idxs.astype(int), adata.var.index.get_loc(split[1])]
    c = counts[inters_idxs.astype(int), adata.var.index.get_loc(split[2])]
    
    return [inters_idxs.astype(int), np.mean(np.array([a,b,c]), axis=0)]

def pos_pos_neg(adata, split):
    counts = adata.raw.X.A
    
    inters_idxs = np.array(list(set(np.where(counts[:,adata.var.index.get_loc(split[0])]>0)[0]) & set(np.where(counts[:,adata.var.index.get_loc(split[1])]>0)[0]) & set(np.where(counts[:,adata.var.index.get_loc(split[2])]==0)[0])))    
    
    a = counts[inters_idxs.astype(int), adata.var.index.get_loc(split[0])]
    b = counts[inters_idxs.astype(int), adata.var.index.get_loc(split[1])]
    c = counts[inters_idxs.astype(int), adata.var.index.get_loc(split[2])]


    return [inters_idxs.astype(int), np.mean(np.array([a,b,c]), axis=0)]

#Find HVGs using Poisson Gamma Negative Binomial Model
def meanCVfit(
    adata,
    reads_use = False,
    diffCV_cutoff = 0.15,
    do_spike = False
    ):

    """\
        Calculate highly variable genes according to Poission-Gamma Negative Binomial model

        Parameters
        ----------
        adata
            AnnData object with adata.X corresponding to the raw counts (not normalized, log-transformed or scaled)
        -------
    """

    mean_emp = np.mean(adata.layers['raw'].A, axis = 0)
    var_emp = np.var(adata.layers['raw'].A, axis = 0)
    cv_emp = np.sqrt(var_emp)/mean_emp

    row_sums = np.sum(adata.layers['raw'].A, axis = 1)
    size_factor = row_sums/np.mean(row_sums)
    fit_alpha, fit_loc, fit_scale = sp.stats.gamma.fit(data = size_factor, floc = 0)

    if do_spike == True:
        spike_genes_index = ['^ERCC' in i for i in list(adata.var.index)]
        spike_genes = list(adata.var.index[spike_genes_index])
    
    fig, axes = plt.subplots(1,1)
    sns.distplot(a = size_factor, bins = 50)
    if reads_use == False:
        axes.set(xlabel = r'$\frac{N_{UMI}}{<N_{UMI}>}$', xlim = [0, np.quantile(size_factor,0.999)])
    else:
        axes.set(xlabel = r'$\frac{N_{Reads}}{<N_{Reads}>}$', xlim = [0, np.quantile(size_factor,0.999)])
    fig.tight_layout()
    plt.show()

    fig, axes = plt.subplots(1,1)
    plt.plot(size_factor, sp.stats.gamma.pdf(x = size_factor, a = fit_alpha, scale = fit_scale), 'o', color = 'red')
    axes.set(title = 'Gamma dist for size factor', xlim = [0, np.quantile(size_factor,0.999)])
    fig.tight_layout()
    plt.show()

    print(f'shape = {np.around(fit_alpha, 2)}')
    print(f'rate = {np.around(1/fit_scale, 2)}')

    a_i = np.repeat(fit_alpha, len(mean_emp))
    b_i = (1/fit_scale)/mean_emp
    mean_NB = a_i/b_i
    var_NB = a_i*(1+b_i)/(b_i**2)
    cv_NB = np.sqrt(var_NB)/mean_NB
    diffCV = np.log(cv_emp) - np.log(cv_NB)

    pass_cutoff = [i for i in list(adata.var.index[np.where(diffCV > diffCV_cutoff, True, False)]) if i in list(adata.var.index[np.where((mean_emp > 0.001) & (mean_emp < 5), True, False)])]

    fig, axes = plt.subplots(1,1)
    plt.plot(mean_emp, cv_emp, 'o', color = 'grey', markersize = 1, label = 'other genes')
    mean_emp_pass_cutoff = []
    cv_emp_pass_cutoff = []
    for i in pass_cutoff:
        mean_emp_pass_cutoff.append(mean_emp[np.where(adata.var.index == i)[0][0]])
        cv_emp_pass_cutoff.append(cv_emp[np.where(adata.var.index == i)[0][0]])
    plt.plot(mean_emp_pass_cutoff, cv_emp_pass_cutoff, 'o', color = 'black', markersize = 1, label = 'highly variable genes')
    if do_spike == True:
        mean_emp_spiked = []
        cv_emp_spiked = []
        for i in spike_genes:
            mean_emp_spiked.append(mean_emp[np.where(adata.var.index == i)[0][0]])
            cv_emp_spiked.append(cv_emp[np.where(adata.var.index == i)[0][0]])
        plt.plot(mean_emp_spiked, cv_emp_spiked, '^', color = 'red', markersize = 1)
    plt.plot(np.linspace(np.min(mean_emp), np.max(mean_emp)), np.sqrt(1/(np.linspace(np.min(mean_emp), np.max(mean_emp)))), linestyle = '--', linewidth = 2, color = 'red')
    plt.plot(mean_NB[np.argsort(mean_NB)], cv_NB[np.argsort(mean_NB)], color = 'magenta', linewidth = 2)
    axes.set(xlabel = 'Mean Counts', ylabel = 'CV (counts)', xscale='log', yscale='log')
    plt.legend(prop={'size': 11})
    fig.tight_layout()
    plt.show()
    
    
    hvg_bool = []
    for i in range(adata.shape[1]):
        if (adata.var.index[i] in pass_cutoff): hvg_bool.append(True)
        else: hvg_bool.append(False)
    
    return hvg_bool

#Find genes that are enriched for each cluster
def ClusterSpecificGenes(adata, genes, obs): #Use obs = 'Type_num' for atlas and obs = 'Type_iGB' for all other time points
  all_var_genes = genes
  percentages = np.zeros((len(all_var_genes),len(adata.obs[obs].values.categories)))
  all_var_genes_index = []
  for i in all_var_genes:
    all_var_genes_index.append(np.where(adata.var.index.values == i)[0][0])
  clusters = list(adata.obs[obs].values.categories)
  for index, value in enumerate(clusters):
    cells_in_clust = adata.obs.index[adata.obs[obs].values == value]
    cells_in_clust_index = []
    for i in cells_in_clust:
      cells_in_clust_index.append(np.where(adata.obs.index.values == i)[0][0])
    percentages[:,index] = adata.layers['raw'][cells_in_clust_index,:][:, all_var_genes_index].getnnz(axis = 0)/len(cells_in_clust_index)
  var_genes = []
  var_genes_index = []
  for i in range(len(all_var_genes_index)):
    if any(i > 0.3 for i in percentages[i,:]) == True:
      var_genes.append(all_var_genes[i])
      var_genes_index.append(all_var_genes_index[i])
  X = adata.layers['raw'].copy()
  counts_per_cell = X.sum(1)
  counts_per_cell = np.ravel(counts_per_cell)
  counts = np.asarray(counts_per_cell)
  after = np.median(counts[counts>0], axis=0)
  counts += (counts == 0)
  counts = counts / after
  sparsefuncs.inplace_row_scale(X, 1/counts)
  E = np.zeros((len(var_genes),len(adata.obs[obs].values.categories)))
  for index, value in enumerate(clusters):
    cells_in_clust = adata.obs.index[adata.obs[obs].values == value]
    cells_in_clust_index = []
    for i in cells_in_clust:
      cells_in_clust_index.append(np.where(adata.obs.index.values == i)[0][0])
    E[:,index] = np.log(X[cells_in_clust_index,:][:, var_genes_index].mean(axis = 0)+1)
  a = np.zeros(len(var_genes))
  for i in range(len(a)):
    ranking_E = np.sort(E[i,:])
    a[i] = np.mean(ranking_E[-7:])/np.mean(ranking_E[:7])
  to_return = list(np.array(var_genes)[a>8])

  return to_return

def pre_step1(adata):
    adata.var['highly_variable'] = meanCVfit(adata)
    
    adata.raw = adata

    sc.pp.scale(adata, max_value=10) #scale
    sc.tl.pca(adata, svd_solver='arpack') #run PCA

    Z = harmonize(adata.obsm['X_pca'], adata.obs, batch_key = 'Batch')
    adata.obsm['X_harmony'] = Z

    #need these b/c will re-run kNN in UMAP 2D space
    sc.pp.neighbors(adata, n_neighbors=25, use_rep='X_harmony')
    sc.tl.umap(adata)
    
    return adata

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
        if len(train_temp) < 200:
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
        if len(train_temp) < 200:
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
    yticks = np.array(range(diagcm.shape[0])) + 1
    dot_ax.set_yticklabels(yticks)
    x_ticks = range(diagcm.shape[1])
    dot_ax.set_xticks(x_ticks)
    xticks = np.array(range(diagcm.shape[1])) + 1
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

def nn_voting(adata):
    adata.obs['idx'] = np.arange(adata.shape[0])

    #re-make graph using UMAP space for voting below
    sc.pp.neighbors(adata, n_neighbors=15, use_rep='X_umap')
    sc.tl.umap(adata)

    adata_un_idx = adata[adata.obs['Type_iGB']=='Unassigned',:].obs.idx.values #indices of unassigned cells
    neigh_matx = adata.obsp['distances'].A #each row's nonzero entries tells neighbors
    adata_iGBs = list(adata.obs['Type_iGB'].values) #all the step1 assignments
    n_neighbs = adata.uns['neighbors']['params']['n_neighbors']
    print(adata.uns['neighbors']['params'])
    print(' ')
    
    print("Pre-step 2 unassigned: ", adata_iGBs.count('Unassigned'),
          adata_iGBs.count('Unassigned')/adata.shape[0])
    
    
    adata.uns['Unassigned info'] = {'after step 1': [], 'after step 2': []}
    
    adata.uns['Unassigned info']['after step 1'] = [adata_iGBs.count('Unassigned'), 
                                                    adata_iGBs.count('Unassigned')/adata.shape[0]]
    delta = 1 #init delta
    while (delta>0.005):    
        un_frac1 = adata_iGBs.count('Unassigned')/adata.shape[0] #frac of unassigned cells before voting

        #loop thru each cell
        for i in adata_un_idx:

            #so that it only loops thru still-unassigned cells after first pass
            if (adata.obs.Type_iGB[i]=='Unassigned'):
                neighbs_idx = np.where(neigh_matx[i,:]>0)[0] #cell i's neighbors
                neighbs_iGBs = adata[neighbs_idx].obs['Type_iGB'] #the neighbors' iGBs

                #if there's a type in the neighbors that is majority, assign it
                if (neighbs_iGBs.value_counts()[0] > n_neighbs/2):
                    adata_iGBs[i] = neighbs_iGBs.value_counts().index[0]

                #update the IDs to help assignment of next cell
                adata.obs['Type_iGB'] =  pd.Categorical(adata_iGBs)

        un_frac2 = adata_iGBs.count('Unassigned')/adata.shape[0] #frac of unassigned cells after voting

        delta = (un_frac1-un_frac2)/un_frac1   #stop when this changes by less than 0.5%
        print(delta, un_frac2)

    print("Post-step2 unassigned: ", adata_iGBs.count('Unassigned'),
          adata_iGBs.count('Unassigned')/adata.shape[0])
    
    adata.uns['Unassigned info']['after step 2'] = [adata_iGBs.count('Unassigned'), 
                                                    adata_iGBs.count('Unassigned')/adata.shape[0]]

    return adata

#canpy's subsample function to sample w/replacement
def subsample(
    data: Union[AnnData, np.ndarray, spmatrix],
    fraction: Optional[float] = None,
    n_obs: Optional[int] = None,
    random_state=0,
    copy: bool = False,
    replace: bool = False) -> Optional[AnnData]:
    """\
    Subsample to a fraction of the number of observations.
    Parameters
    ----------
    data
        The (annotated) data matrix of shape `n_obs` × `n_vars`.
        Rows correspond to cells and columns to genes.
    fraction
        Subsample to this `fraction` of the number of observations.
    n_obs
        Subsample to this number of observations.
    random_state
        Random seed to change subsampling.
    copy
        If an :class:`~anndata.AnnData` is passed,
        determines whether a copy is returned.
    Returns
    -------
    Returns `X[obs_indices], obs_indices` if data is array-like, otherwise
    subsamples the passed :class:`~anndata.AnnData` (`copy == False`) or
    returns a subsampled copy of it (`copy == True`).
    """
    np.random.seed(random_state)
    old_n_obs = data.n_obs if isinstance(data, AnnData) else data.shape[0]
    if n_obs is not None:
        new_n_obs = n_obs
    elif fraction is not None:
        if fraction > 1 or fraction < 0:
            raise ValueError(
                f'`fraction` needs to be within [0, 1], not {fraction}'
            )
        new_n_obs = int(fraction * old_n_obs)
       # logg.debug(f'... subsampled to {new_n_obs} data points')
    else:
        raise ValueError('Either pass `n_obs` or `fraction`.')
    obs_indices = np.random.choice(old_n_obs, size=new_n_obs, replace=replace)
    if isinstance(data, AnnData):
        if copy:
            return data[obs_indices].copy()
        else:
            data._inplace_subset_obs(obs_indices)
    else:
        X = data
        return X[obs_indices], obs_indices

def DE(adata, obs_id, obs_id_test, ref):

    sc.tl.rank_genes_groups(adata, groupby=obs_id, groups=[obs_id_test], 
                            reference=ref, method='wilcoxon', n_genes=100)

    wilcLF = adata.uns['rank_genes_groups']['logfoldchanges'].astype([(obs_id_test, '<f8')]).view('<f8') #log fold changes ordered by score

    wilcGenes_s = list(adata.uns['rank_genes_groups']['names'].astype([(obs_id_test, '<U50')]).view('<U50'))  #list of genes ordered by wilc score
    wilcLF_s = adata.uns['rank_genes_groups']['logfoldchanges'].astype([(obs_id_test, '<f8')]).view('<f8') #numpy array of logfoldchnages to be ordered lowest to highest
    wilcLF_s.sort() #log fold changes from lowest to highest
    wilcGenes = [] #list of genes ordered by logfold change: low to high

    for i in wilcLF_s:
        gene_idx = np.where(wilcLF == i)[0][0]
        wilcGenes.append(wilcGenes_s[gene_idx])

    wilcGenes_correct = []
    for i in reversed(wilcGenes):
        wilcGenes_correct.append(i) # #list of genes ordered by logfold change high to low
        
    return wilcGenes_correct
    
