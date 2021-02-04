def step2(adata):
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
    
    print("Start unassigned: ", adata_iGBs.count('Unassigned'))

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

    print("End unassigned: ", adata_iGBs.count('Unassigned'))

    return adata
