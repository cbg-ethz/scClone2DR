from ..utils import *
import torch
import pandas as pd
import numpy as np
from .fastdrug import FastDrug
from .rnadata import RNAData
from .basedataset import BaseDataset
import os
import copy

class RealData(BaseDataset):

    def __init__(self, path_fastdrug=None, path_rna=None, path_info_cohort=None, concentration_drug=None):
        super(BaseDataset, self).__init__()
        self.RNA = RNAData(path_rna)
        self.FD = FastDrug(path_fastdrug, samples = self.RNA.sample_names, concentration_drug = concentration_drug)
        sample_names = np.intersect1d(self.RNA.sample_names, self.FD.sample_names)
        self.sample_names = sample_names
        self.drugs = self.FD.selected_drugs
        if not(path_rna is None):
            self.dfclones = pd.read_csv(self.RNA.path_rna+'clone_infos.csv', index_col=0)
            self.Kmax = self.dfclones.shape[0]
            df = self.RNA.load_data(self.RNA.sample_names[0])
            dims = [col for col in df.columns if 'dim' in col]
            self.latent_dim = len(dims)
            if (dims[0]).count('_')>=2:
                self.feature_names = [item.split('_', 2)[2] for item in dims] 
            else:
                self.feature_names = dims
            self.cluster2clonelabel = []
            self.clonelabel2cat = {}
            for index, row in self.dfclones.iterrows():
                self.cluster2clonelabel.append(row['clonelabel'])
                self.clonelabel2cat[row['clonelabel']] = row['clonecategory']
            self._init_cat_clonelabel()
          
        self.dfinfo = None
        if not(path_info_cohort is None):
            dfinfo = pd.read_csv(path_info_cohort, sep='\t')
            # Filtering patients kept for the analysis
            dfinfo['sampleID'] = dfinfo['sampleID'].apply(lambda x: self.RNA.get_sampleID_from_file(x))
            dfinfo = dfinfo[dfinfo['sampleID'].apply(lambda x: x in self.RNA.sample_names).values]
            self.dfinfo = dfinfo.set_index('sampleID')
            unique_patient_ids = self.dfinfo['patient_id'].unique()
            patient_id_mapping = {patient_id: idx for idx, patient_id in enumerate(unique_patient_ids)}
            self.dfinfo['patient_id'] = self.dfinfo['patient_id'].map(patient_id_mapping)
            self.dfinfo = self.dfinfo.reindex(self.RNA.sample_names)

    
    def get_real_data(self, concentration_DMSO='100', concentration_drug='10', standardize=True, seed=1, test_size=0.2, get_random_split=False):
        sample_names = self.sample_names
        print('Total number of samples: ', len(sample_names))

        ########### LOAD drug data
        self.FD.load_FD_data(self.sample_names, concentration_DMSO=concentration_DMSO, concentration_drug=concentration_drug)
        self.drugs = self.FD.selected_drugs

        ########## LOAD RNA DATA: Number of cells per subclone
        max_num_cells_tmp = 50000
        N = len(self.sample_names)
        Kmax = self.RNA.Kmax
        latent_dim = self.latent_dim

        # Lists to store sparse tensor components
        indices = []  # Nonzero indices (cloneID, patientID, cellID, feature_dim)
        values = []   # Corresponding feature values

        sample2cloneID2nb_cells_rna = {}
        max_num_cells = 0

        for id_patient, sample in enumerate(sample_names):
            cloneID2nb_cells_rna = []
            for cloneID in range(Kmax):
                idx_data_clone, features_clone = self.RNA.get_features(sample, cloneID=cloneID)

                if features_clone.shape[0] > 0:  # Avoid storing empty data
                    #features_clone = torch.tensor(features_clone, dtype=torch.float16)  # Convert to float16

                    # Store indices and values in sparse format
                    for i, idx in enumerate(idx_data_clone):
                        for j in range(latent_dim):
                            indices.append([cloneID, id_patient, i, j])
                            values.append(features_clone[i, j])

                    cloneID2nb_cells_rna.append(features_clone.shape[0])
                    max_num_cells = max(max_num_cells, features_clone.shape[0])
                else:
                    cloneID2nb_cells_rna.append(0)

            sample2cloneID2nb_cells_rna[sample] = cloneID2nb_cells_rna

        # Convert lists to tensors
        indices = torch.tensor(indices, dtype=torch.long).t()  # Transpose to (4, num_entries)
        print(type(values[0]))
        values = torch.tensor(values).float()  # Use float16 to reduce memory
        # If standardization is needed, convert to dense, compute, and convert back to sparse
        if standardize:
            S_dense = torch.full((Kmax, N, max_num_cells, latent_dim), float('nan'))  # Initialize with NaNs
            S_dense[indices[0,:], indices[1,:], indices[2,:], indices[3,:]] = values

            meanS = torch.nanmean(S_dense, dim=(0,1,2))
            def nanstd(x): 
                return torch.sqrt(torch.nanmean(torch.pow(x - torch.nanmean(x, dim=(0,1,2))[None,None,None,:], 2), dim=(0,1,2)))
            stdS = nanstd(S_dense)
            S_dense = (S_dense - meanS[None,None,None,:]) / stdS[None,None,None,:]

        ########## Building the global data dictionary
        C = np.max([len(nb_cells_c) for nb_cells_c in self.FD.sample2nb_cells_per_well_control.values()])
        R = np.max([len(nb_cells_c) for nb_cells_c in self.FD.sample2nb_cells_per_well.values()])
        selected_drugs = self.FD.selected_drugs
        self.drugs = selected_drugs
        D = len(selected_drugs)

        data = {}

        data['n0_r'] = torch.zeros((R,D,N))
        data['n_r'] = torch.zeros((R,D,N))
        data['n0_c'] = torch.zeros((C,N))
        data['n_c'] = torch.zeros((C,N))
        data['n_rna'] = torch.zeros((Kmax,N))
        data['N'], data['Kmax'], data['C'], data['R'], data['D'] = N, Kmax, C, R, D
        data['masks'] = {}

        data['masks']['SingleCell'] = torch.tensor(~torch.isnan(S_dense[:,:,:,0])).clone().detach()

        # Use sparse tensor for data
        data['X'] = torch.nan_to_num(S_dense)

        logscores = np.zeros((D,N))
        for ids, sample in enumerate(self.sample_names):
            for cloneID, nb_cells_rna in enumerate(sample2cloneID2nb_cells_rna[sample]):
                data['n_rna'][cloneID,ids] = int(nb_cells_rna)
            ls_c = torch.tensor(self.FD.sample2nb_cells_per_well_control[sample]).int()
            data['n_c'][:len(ls_c),ids] = ls_c.clone()
            ls0_c = torch.tensor(self.FD.sample2tumor_control[sample]).int()
            data['n0_c'][:len(ls0_c),ids] = (ls_c - ls0_c).clone()
            frac_c = 1. - torch.mean(data['n0_c'][:len(ls0_c),ids]/data['n_c'][:len(ls_c),ids])
            for iddrug, drug in enumerate(selected_drugs):
                ls_r = torch.tensor(self.FD.sample2nb_cells_per_well[(drug,sample)]).int()
                data['n_r'][:len(ls_r),iddrug,ids] = ls_r.clone()
                ls0_r = torch.tensor(self.FD.sample2tumor[(drug,sample)]).int()
                data['n0_r'][:len(ls0_r),iddrug,ids] = (ls_r-ls0_r).clone()
                frac_r = 1. - torch.mean(data['n0_r'][:len(ls0_r),iddrug,ids]/data['n_r'][:len(ls_r),iddrug,ids])
                logscores[iddrug,ids] = abs(torch.log(frac_c / frac_r))

        data['ini_proportions'] = get_ini_proportions(data)

        ########## Splitting the data in train and test.



        self.Kmax, self.N, self.R, self.C, self.D = Kmax, N, R, C, D

        data = self.add_masks(data)
        data['simulated_data'] = False
        data['theta_rna'] = None
        data['single_cell_features'] = True

        data, _ = self.add_design_preassay(data, {}, sample_names, [])

        self.data = data
#        self.data, _, _, _ = self.get_real_data_split([i for i in range(self.N)], [])

        if get_random_split:
            np.random.seed(seed)
            idxs_test = np.random.choice(list(range(N)), size=int(N*test_size), replace=False)
            idxs_train = np.array([i for i in range(N) if not(i in idxs_test)])
            return self.get_real_data_split(idxs_train, idxs_test)
        else:
            return self.data

    def add_masks(self, data):
        N = data['N']
        data['masks']['RNA'] = torch.ones((self.Kmax,N), dtype=torch.bool)
        for i in range(N):
            for k in range(self.Kmax):
                if data['n_rna'][k,i]<=0.5:
                    data['masks']['RNA'][k,i] = 0
        
        data['masks']['C'] = torch.ones((self.C,N), dtype=torch.bool)
        for i in range(N):
            sample = self.sample_names[i]
            size = len(self.FD.sample2nb_cells_per_well_control[sample])
            if size<=self.C-1:
                ls = [c for c in range(size,self.C)]
                for c in ls:
                    data['masks']['C'][c,i] = 0
        
        data['masks']['R'] = torch.ones((self.R,self.D,N), dtype=torch.bool)
        for idxdrug, drug in enumerate(self.FD.selected_drugs):
            for i in range(N):
                sample = self.sample_names[i]
                size = len(self.FD.sample2nb_cells_per_well[(drug,sample)])
                if size<=self.R-1:
                    ls = [r for r in range(size,self.R)]
                    for r in ls:
                        data['masks']['R'][r,idxdrug,i] = 0
        return data
    
    def get_real_data_split(self, idxs_train, idxs_test):
        Ntrain = len(idxs_train)
        Ntest = len(idxs_test)
        Ntot = Ntrain + Ntest
        data_train = {'X': self.data['X'][:,idxs_train,:,:], 'D':self.D, 'R':self.R, 'C':self.C, 'Kmax':self.Kmax, 'N':Ntot}
        data_train['masks'] = {}
        data_train['masks']['SingleCell'] = self.data['masks']['SingleCell'][:,idxs_train,:]
        data_train['masks']['RNA'] = torch.cat((self.data['masks']['RNA'][:,idxs_train],self.data['masks']['RNA'][:,idxs_test]), dim=1)
        data_train['masks']['C'] = torch.cat((self.data['masks']['C'][:,idxs_train],self.data['masks']['C'][:,idxs_test]), dim=1)
        data_train['masks']['R'] = self.data['masks']['R'][:,:,idxs_train]

        data_train['X_nu_control'] = torch.cat((self.data['X_nu_control'][:,idxs_train,:],self.data['X_nu_control'][:,idxs_test,:]), dim=1)
        data_train['X_nu_drug'] = self.data['X_nu_drug'][:,:,idxs_train,:]

        
        data_train['n0_c'] = torch.zeros((self.C,Ntot))
        data_train['n0_c'][:,:Ntrain] = self.data['n0_c'][:,idxs_train]
        data_train['n0_c'][:,Ntrain:] = self.data['n0_c'][:,idxs_test]

        data_train['n_c'] = torch.zeros((self.C,Ntot))
        data_train['n_c'][:,:Ntrain] = self.data['n_c'][:,idxs_train]
        data_train['n_c'][:,Ntrain:] = self.data['n_c'][:,idxs_test]

        if self.data['n_rna'] is not None:
            data_train['n_rna'] = torch.zeros((self.Kmax,Ntot))
            data_train['n_rna'][:,:Ntrain] = self.data['n_rna'][:,idxs_train]
            data_train['n_rna'][:,Ntrain:] = self.data['n_rna'][:,idxs_test]
        else:
             data_train['n_rna'] = None
            
        data_train['n0_r'] = self.data['n0_r'][:,:,idxs_train]
        data_train['n_r']  = self.data['n_r'][:,:,idxs_train]
        data_train['ini_proportions'] = torch.zeros((Ntot,self.Kmax)) 
        data_train['ini_proportions'][:Ntrain,:] = self.data['ini_proportions'][idxs_train,:]
        data_train['ini_proportions'][Ntrain:,:] = self.data['ini_proportions'][idxs_test,:]

        frac_r = 1. - data_train['n0_r']/data_train['n_r']
        frac_c = 1. - data_train['n0_c']/data_train['n_c']
        data_train['frac_r'] = torch.nan_to_num(frac_r) 
        data_train['frac_c'] = torch.nan_to_num(frac_c) 

        frac_mean_r = torch.sum(data_train['masks']['R']*torch.nan_to_num(frac_r), dim=0) / torch.sum(data_train['masks']['R'], dim=0) 
        frac_mean_c = torch.sum(data_train['masks']['C']*torch.nan_to_num(frac_c), dim=0) / torch.sum(data_train['masks']['C'], dim=0) 

        R, D, N = self.data['n_r'].shape
        C, N = self.data['n_c'].shape
        Kmax, N = self.data['masks']['RNA'].shape

        data_test = {'R':self.R, 'N':Ntest, 'D': self.D, 'C':self.C, 'Kmax':self.Kmax}
        data_test['X_nu_control'] = self.data['X_nu_control'][:,idxs_test,:]
        data_test['X_nu_drug'] = self.data['X_nu_drug'][:,:,idxs_test,:]

        data_test['masks'] = {}
        data_test['masks']['SingleCell'] = self.data['masks']['SingleCell'][:,idxs_test,:]

        data_test['masks']['RNA'] = self.data['masks']['RNA'][:,idxs_test]
        data_test['masks']['C'] = self.data['masks']['C'][:,idxs_test]
        data_test['masks']['R'] = self.data['masks']['R'][:,:,idxs_test]
        data_test['X'] = self.data['X'][:,idxs_test,:,:]
        
        if self.data['n_rna'] is not None:
            data_test['n_rna'] = self.data['n_rna'][:,idxs_test]
        else:
            data_test['n_rna'] = None
        data_test['n0_c'] = self.data['n0_c'][:,idxs_test]
        data_test['n_c']  = self.data['n_c'][:,idxs_test]
        data_test['n0_r'] = self.data['n0_r'][:,:,idxs_test]
        data_test['n_r']  = self.data['n_r'][:,:,idxs_test]
        data_test['ini_proportions'] = self.data['ini_proportions'][idxs_test,:]

        frac_r = 1. - data_test['n0_r']/data_test['n_r']
        frac_c = 1. - data_test['n0_c']/data_test['n_c']
        data_test['frac_r'] = torch.nan_to_num(frac_r) 
        data_test['frac_c'] = torch.nan_to_num(frac_c) 

        frac_mean_r = torch.sum(data_test['masks']['R']*torch.nan_to_num(frac_r), dim=0) / torch.sum(data_test['masks']['R'], dim=0) 
        frac_mean_c = torch.sum(data_test['masks']['C']*torch.nan_to_num(frac_c), dim=0) / torch.sum(data_test['masks']['C'], dim=0) 

        data_test['log_scores'] = torch.log(frac_mean_c.unsqueeze(0) / frac_mean_r)

        data_train['simulated_data'] = False
        data_test['simulated_data'] = False
        data_train['theta_rna'] = None
        data_test['theta_rna'] = None
        data_train['single_cell_features'] = True
        data_test['single_cell_features'] = True
        
        sample_names_train = self.sample_names[idxs_train]
        sample_names_test = self.sample_names[idxs_test]
        
        return data_train, data_test, sample_names_train, sample_names_test

    def add_design_preassay(self, data_train, data_test, sample_names, sample_names_test):

        #### SPLINES

        # FEATURES WELL X kronecker product

        from scipy.interpolate import BSpline
        import skfda


        def get_spline_basis(x, degree=3, nknots=10):
            k = nknots
            knots = np.linspace(0, 1, k - degree)
            bsplines = skfda.representation.basis.BSplineBasis(knots=knots, order=degree + 1)
            res = np.zeros((len(x),k-1))
            for i in range(k-1):
                res[:,i] = bsplines.to_basis().tolist()[i](x).reshape(-1)
            return res.reshape(-1)

        def get_penalty_matrix_splines(degree=3, k=10):
            knots = np.linspace(0, 1, k - degree)
            bsplines = skfda.representation.basis.BSplineBasis(knots=knots, order=degree + 1)
            # calculate penalty matrix
            diffMatrixOrder = 2 
            operator = skfda.misc.operators.LinearDifferentialOperator(diffMatrixOrder)
            regularization = skfda.misc.regularization.L2Regularization(operator)
            penalty_matrix = regularization.penalty_matrix(bsplines)
            return penalty_matrix

        nh_rna = torch.sum(data_train['n_rna'][self.cat2clusters['healthy'],:], dim=0)
        nt_rna = torch.sum(data_train['n_rna'][self.cat2clusters['tumor'],:], dim=0)
        frac_rna = nh_rna / (nh_rna + nt_rna)

        MAX_nc = data_train['n_c'].reshape(-1).max()

        features = ['frac_rna', 'density', 'intercept_pipet', 'intercept_pipet', 'intercept'] 
        degree = 3
        nknots = 5
        feat_dim = nknots-1
        feat_dim_global = 1 + (len(features)-1)*feat_dim
        X = np.zeros((len(sample_names)+len(sample_names_test), data_train['C'], feat_dim_global))
        Xdrug = np.zeros((len(sample_names), data_train['D'], data_train['R'], feat_dim_global))

        for idsample, sample in enumerate(sample_names):
            feat_vec = []
            feat_vec.append(frac_rna[idsample])
            wellpos = self.FD.sample2control_wellpos[sample]
            for i in range(wellpos.shape[0]):
                well1 = wellpos[i,0]-1
                well2 = wellpos[i,1]-1
                feat_vec2 = list(feat_vec)
                feat_vec2 += [data_train['n_c'][i, idsample] / MAX_nc]
                feat_vec2 += [well2%2 * well2/24, (well2+1)%2 * well2/24]
                feat_vec2 = list(get_spline_basis(feat_vec2, degree=degree, nknots=nknots))
                feat_vec2 += [1]
                X[idsample,i,:] = feat_vec2

            for d,drug in enumerate(self.FD.selected_drugs):
                wellpos = self.FD.sample2wellpos[(drug,sample)]
                for i in range(wellpos.shape[0]):
                    well1 = wellpos[i,0]-1
                    well2 = wellpos[i,1]-1
                    feat_vec2 = list(feat_vec)
                    feat_vec2 += [data_train['n_c'][i, idsample] / MAX_nc]
                    feat_vec2 += [well2%2 * well2/24, (well2+1)%2 * well2/24]
                    feat_vec2 = list(get_spline_basis(feat_vec2, degree=degree, nknots=nknots))
                    feat_vec2 += [ 1]
                    Xdrug[idsample,d,i,:] = feat_vec2

        for idsample_bis, sample in enumerate(sample_names_test):
            idsample = idsample_bis + len(sample_names)
            feat_vec = []
            feat_vec.append(frac_rna[idsample])
            wellpos = self.FD.sample2control_wellpos[sample]
            for i in range(wellpos.shape[0]):
                well1 = wellpos[i,0]-1
                well2 = wellpos[i,1]-1
                feat_vec2 = list(feat_vec)
                feat_vec2 += [data_train['n_c'][i, idsample] / MAX_nc]
                feat_vec2 += [well2%2 * well2/24, (well2+1)%2 * well2/24]
                feat_vec2 = list(get_spline_basis(feat_vec2, degree=degree, nknots=nknots))
                feat_vec2 += [1]
                X[idsample,i,:] = feat_vec2

        # FEATURES WELL X kronecker product
        if len(sample_names_test)!=0:            
            nt_rna_test = torch.sum(data_test['n_rna'][self.cat2clusters['tumor'],:], dim=0)
            frac_rna_test = nh_rna_test / (nh_rna_test + nt_rna_test)

            X_test = np.zeros((len(sample_names_test), data_test['C'], feat_dim_global))
            X_testdrug = np.zeros((len(sample_names_test), data_test['D'], data_test['R'], feat_dim_global))



            for idsample, sample in enumerate(sample_names_test):
                feat_vec = []
                feat_vec.append(frac_rna[idsample+len(sample_names)])
                wellpos = self.FD.sample2control_wellpos[sample]
                for i in range(wellpos.shape[0]):
                    well1 = wellpos[i,0]-1
                    well2 = wellpos[i,1]-1
                    feat_vec2 = list(feat_vec)
                    feat_vec2 += [data_train['n_c'][i, idsample] / MAX_nc]
                    feat_vec2 += [well2%2 * well2/24, (well2+1)%2 * well2/24]
                    feat_vec2 = list(get_spline_basis(feat_vec2, degree=degree, nknots=nknots))
                    feat_vec2 += [1]
                    X_test[idsample,i,:] = feat_vec2

                for d,drug in enumerate(self.FD.selected_drugs):
                    wellpos = self.FD.sample2wellpos[(drug,sample)]
                    for i in range(wellpos.shape[0]):
                        well1 = wellpos[i,0]-1
                        well2 = wellpos[i,1]-1
                        feat_vec2 = list(feat_vec)
                        feat_vec2 += [data_train['n_c'][i, idsample] / MAX_nc]
                        feat_vec2 += [well2%2 * well2/24, (well2+1)%2 * well2/24]
                        feat_vec2 = list(get_spline_basis(feat_vec2, degree=degree, nknots=nknots))
                        feat_vec2 += [1]
                        X_testdrug[idsample,d,i,:] = feat_vec2

        # for j in range(len(features)):
        #     X[:,:,j] = X[:,:,j] / np.sum(X[:,:,j]) #(X[:,j]-np.mean(X[:,j]))/np.std(X[:,j])
        #     X_test[:,:,j] = X_test[:,:,j] / np.sum(X[:,:,j]) #(X[:,j]-np.mean(X[:,j]))/np.std(X[:,j])
        #     Xdrug[:,:,:,j] = Xdrug[:,:,:,j] / np.sum(Xdrug[:,:,:,j]) #(X[:,j]-np.mean(X[:,j]))/np.std(X[:,j])
        #     X_testdrug[:,:,:,j] = X_testdrug[:,:,:,j] / np.sum(Xdrug[:,:,:,j]) #(X[:,j]-np.mean(X[:,j]))/np.std(X[:,j])

        data_train['X_nu_control'] = torch.tensor(X).float().permute(1,0,2)
        #data['X_nu_control'] = torch.tensor(X).float().permute(1,0,2)
        data_train['X_nu_drug'] = torch.tensor(Xdrug).float().permute(2,1,0,3)

        if len(sample_names_test)!=0:
            data_test['X_nu_control'] = torch.tensor(X_test).float().permute(1,0,2)

            data_test['X_nu_drug'] = torch.tensor(X_testdrug).float().permute(2,1,0,3)

            #data['X_nu_drug'] = torch.cat((torch.tensor(Xdrug),torch.tensor(X_testdrug)), 0).float().permute(2,1,0,3)

        from scipy.linalg import block_diag
        Ssplines = np.kron(np.eye(len(features)-1,dtype=int),get_penalty_matrix_splines(degree=degree, k=nknots))

        self.Ssplines = torch.tensor(block_diag(Ssplines, np.zeros((1,1)))).float()

        return data_train, data_test  
        
        
        
        
    def get_bulk_from_real_data(self, dic, cluster2cat, cat2clusters):
        data = copy.deepcopy(dic)
                
        # X: Kmax, N, max_num_cells, latent_dim
        Z = torch.zeros((2,data['X'].shape[1],data['X'].shape[2],data['X'].shape[3])) # Kmax x N x x nb cells x dim
        Ndrug = data['X'].shape[1]
        props = torch.zeros((data['n_c'].shape[1],2))
        frac = data['n0_c'] / data['n_c']
        props[:,0] = torch.sum(data['masks']['C']*torch.nan_to_num(frac), dim=0) / torch.sum(data['masks']['C'], dim=0) 
        props[:,1] = 1-props[:,0]
        data['ini_proportions'] = copy.deepcopy(props)
        tempweights = data['ini_proportions'].T[:,:Ndrug] # Kmax x N
        weights = torch.zeros((data['X'].shape[0], data['X'].shape[1]))
        temp_mask = torch.zeros((2, data['X'].shape[1], data['X'].shape[2]), dtype=torch.bool)                   
        for i in range(props.shape[0]):
            for k,cat in enumerate(cluster2cat):
                nb_in_cat = torch.sum(data['masks']['RNA'][cat2clusters[cat],i])
                if nb_in_cat==0:
                    weights[k,i] = 0
                else:
                    weights[k,i] = tempweights[int(cat=='tumor'),i] / nb_in_cat
                if cat== "healthy":
                    temp_mask[0,i,:] = (temp_mask[0,i,:] | data['masks']['SingleCell'][k,i,:])
                else:
                    temp_mask[1,i,:] = (temp_mask[1,i,:] | data['masks']['SingleCell'][k,i,:])


        Z[0,:,:,:] = torch.sum(data['X'] * weights[:,:,None,None], dim=0)
        Z[1,:,:,:] = copy.deepcopy(Z[0,:,:,:])
        data['X'] = torch.nan_to_num(copy.deepcopy(Z))

        data['Kmax'] = 2
        self.Kmax = 2
        N = data['n_rna'].shape[1]
        data['n_rna'] =  None
        data['masks']['RNA'] = torch.zeros((2,N), dtype=torch.bool)
        
        data['masks']['SingleCell'] = temp_mask

        self.cluster2clonelabel = ['healthy','tumor']
        self.cluster2cat = ['healthy','tumor']
        self.clonelabel2cat = ['healthy','tumor']
        self.clonlabel2clusters = {'healthy': [0], 'tumor': [1]}
        self.cat2clusters = {'healthy': [0], 'tumor': [1]}
        self.cat2clonelabels = {'healthy': ['healthy'], 'tumor': ['tumor']}
        self.clonelabel2cat = {'healthy':'healthy', 'tumor':'tumor'}
        self._init_cat_clonelabel()
        return data
    
    def add_design_preassay_bulk(self, data_train, data_test, sample_names, sample_names_test):

        #### SPLINES

        # FEATURES WELL X kronecker product

        from scipy.interpolate import BSpline
        import skfda


        def get_spline_basis(x, degree=3, nknots=10):
            k = nknots
            knots = np.linspace(0, 1, k - degree)
            bsplines = skfda.representation.basis.BSplineBasis(knots=knots, order=degree + 1)
            res = np.zeros((len(x),k-1))
            for i in range(k-1):
                res[:,i] = bsplines.to_basis().tolist()[i](x).reshape(-1)
            return res.reshape(-1)

        def get_penalty_matrix_splines(degree=3, k=10):
            knots = np.linspace(0, 1, k - degree)
            bsplines = skfda.representation.basis.BSplineBasis(knots=knots, order=degree + 1)
            # calculate penalty matrix
            diffMatrixOrder = 2 
            operator = skfda.misc.operators.LinearDifferentialOperator(diffMatrixOrder)
            regularization = skfda.misc.regularization.L2Regularization(operator)
            penalty_matrix = regularization.penalty_matrix(bsplines)
            return penalty_matrix


        MAX_nc = data_train['n_c'].reshape(-1).max()

        features = ['density', 'intercept_pipet', 'intercept_pipet', 'intercept'] 
        degree = 3
        nknots = 5
        feat_dim = nknots-1
        feat_dim_global = 1+(len(features)-1)*feat_dim
        X = np.zeros((len(sample_names)+len(sample_names_test), data_train['C'], feat_dim_global))
        Xdrug = np.zeros((len(sample_names), data_train['D'], data_train['R'], feat_dim_global))

        for idsample, sample in enumerate(sample_names):
            feat_vec = []
            wellpos = self.FD.sample2control_wellpos[sample]
            for i in range(wellpos.shape[0]):
                well1 = wellpos[i,0]-1
                well2 = wellpos[i,1]-1
                feat_vec2 = list(feat_vec)
                feat_vec2 += [data_train['n_c'][i, idsample] / MAX_nc]
                feat_vec2 += [well2%2 * well2/24, (well2+1)%2 * well2/24]
                feat_vec2 = list(get_spline_basis(feat_vec2, degree=degree, nknots=nknots))
                feat_vec2 += [1]
                X[idsample,i,:] = feat_vec2

            for d,drug in enumerate(self.FD.selected_drugs):
                wellpos = self.FD.sample2wellpos[(drug,sample)]
                for i in range(wellpos.shape[0]):
                    well1 = wellpos[i,0]-1
                    well2 = wellpos[i,1]-1
                    feat_vec2 = list(feat_vec)
                    feat_vec2 += [data_train['n_c'][i, idsample] / MAX_nc]
                    feat_vec2 += [well2%2 * well2/24, (well2+1)%2 * well2/24]
                    feat_vec2 = list(get_spline_basis(feat_vec2, degree=degree, nknots=nknots))
                    feat_vec2 += [ 1]
                    Xdrug[idsample,d,i,:] = feat_vec2

        for idsample_bis, sample in enumerate(sample_names_test):
            idsample = idsample_bis + len(sample_names)
            feat_vec = []
            wellpos = self.FD.sample2control_wellpos[sample]
            for i in range(wellpos.shape[0]):
                well1 = wellpos[i,0]-1
                well2 = wellpos[i,1]-1
                feat_vec2 = list(feat_vec)
                feat_vec2 += [data_train['n_c'][i, idsample] / MAX_nc]
                feat_vec2 += [well2%2 * well2/24, (well2+1)%2 * well2/24]
                feat_vec2 = list(get_spline_basis(feat_vec2, degree=degree, nknots=nknots))
                feat_vec2 += [1]
                X[idsample,i,:] = feat_vec2

            
        if len(sample_names_test)!=0:            

            X_test = np.zeros((len(sample_names_test), data_test['C'], feat_dim_global))
            X_testdrug = np.zeros((len(sample_names_test), data_test['D'], data_test['R'], feat_dim_global))



            for idsample, sample in enumerate(sample_names_test):
                feat_vec = []
                wellpos = self.FD.sample2control_wellpos[sample]
                for i in range(wellpos.shape[0]):
                    well1 = wellpos[i,0]-1
                    well2 = wellpos[i,1]-1
                    feat_vec2 = list(feat_vec)
                    feat_vec2 += [data_train['n_c'][i, idsample] / MAX_nc]
                    feat_vec2 += [well2%2 * well2/24, (well2+1)%2 * well2/24]
                    feat_vec2 = list(get_spline_basis(feat_vec2, degree=degree, nknots=nknots))
                    feat_vec2 += [1]
                    X_test[idsample,i,:] = feat_vec2

                for d,drug in enumerate(self.FD.selected_drugs):
                    wellpos = self.FD.sample2wellpos[(drug,sample)]
                    for i in range(wellpos.shape[0]):
                        well1 = wellpos[i,0]-1
                        well2 = wellpos[i,1]-1
                        feat_vec2 = list(feat_vec)
                        feat_vec2 += [data_train['n_c'][i, idsample] / MAX_nc]
                        feat_vec2 += [well2%2 * well2/24, (well2+1)%2 * well2/24]
                        feat_vec2 = list(get_spline_basis(feat_vec2, degree=degree, nknots=nknots))
                        feat_vec2 += [1]
                        X_testdrug[idsample,d,i,:] = feat_vec2

        # for j in range(len(features)):
        #     X[:,:,j] = X[:,:,j] / np.sum(X[:,:,j]) #(X[:,j]-np.mean(X[:,j]))/np.std(X[:,j])
        #     X_test[:,:,j] = X_test[:,:,j] / np.sum(X[:,:,j]) #(X[:,j]-np.mean(X[:,j]))/np.std(X[:,j])
        #     Xdrug[:,:,:,j] = Xdrug[:,:,:,j] / np.sum(Xdrug[:,:,:,j]) #(X[:,j]-np.mean(X[:,j]))/np.std(X[:,j])
        #     X_testdrug[:,:,:,j] = X_testdrug[:,:,:,j] / np.sum(Xdrug[:,:,:,j]) #(X[:,j]-np.mean(X[:,j]))/np.std(X[:,j])

        data_train['X_nu_control'] = torch.tensor(X).float().permute(1,0,2)
        #data['X_nu_control'] = torch.tensor(X).float().permute(1,0,2)
        data_train['X_nu_drug'] = torch.tensor(Xdrug).float().permute(2,1,0,3)

        if len(sample_names_test)!=0:
            data_test['X_nu_control'] = torch.tensor(X_test).float().permute(1,0,2)

            data_test['X_nu_drug'] = torch.tensor(X_testdrug).float().permute(2,1,0,3)

            #data['X_nu_drug'] = torch.cat((torch.tensor(Xdrug),torch.tensor(X_testdrug)), 0).float().permute(2,1,0,3)

        from scipy.linalg import block_diag
        Ssplines = np.kron(np.eye(len(features)-1,dtype=int),get_penalty_matrix_splines(degree=degree, k=nknots))

        self.Ssplines = torch.tensor(block_diag(Ssplines, np.zeros((1,1)))).float()

        return data_train, data_test  
    
    
    def get_bimodal_from_real_data(self, dic, cluster2cat, cat2clusters):
        data = copy.deepcopy(dic)
        Z = torch.zeros((2,data['X'].shape[1],data['X'].shape[2],data['X'].shape[3])) # Kmax x N x x nb cells x dim
        Ndrug = data['X'].shape[1]
        props = torch.zeros((data['n_c'].shape[1],2))
        frac = data['n0_c'] / data['n_c']
        props[:,0] = torch.sum(data['masks']['C']*torch.nan_to_num(frac), dim=0) / torch.sum(data['masks']['C'], dim=0) 
        props[:,1] = 1-props[:,0]
        data['ini_proportions'] = copy.deepcopy(props)
        tempweights = data['ini_proportions'].T[:,:Ndrug] # Kmax x N
        weights = torch.zeros((data['X'].shape[0], data['X'].shape[1]))
        for i in range(props.shape[0]):
            for k,cat in enumerate(cluster2cat):
                nb_in_cat = torch.sum(data['masks']['RNA'][cat2clusters[cat],i])
                if nb_in_cat==0:
                    weights[k,i] = 0
                else:
                    weights[k,i] = tempweights[int(cat=='tumor'),i] / nb_in_cat
        weights_tumor = weights[cat2clusters['tumor'],:]
        weights_tumor = weights_tumor / (torch.sum(weights_tumor, dim=0)[None,:])
        Z[1,:,:,:] = torch.sum(data['X'][cat2clusters['tumor'],:,:,:] * weights_tumor[:,:,None,None], dim=0)
        
        weights_healthy = weights[cat2clusters['healthy'],:]
        weights_healthy = weights_healthy / (torch.sum(weights_healthy, dim=0)[None,:])
        Z[0,:,:,:] = torch.sum(data['X'][cat2clusters['healthy'],:,:,:] * weights_healthy[:,:,None,None], dim=0)
        data['X'] = torch.nan_to_num(copy.deepcopy(Z))
        
        
        data['Kmax'] = 2
        self.Kmax = 2
        temp_rna = torch.zeros((2, data['n_rna'].shape[1]))
        temp_mask = torch.zeros((2, data['n_rna'].shape[1]), dtype=torch.bool)
        temp_mask_sc = torch.zeros((2, data['X'].shape[1], data['X'].shape[2]), dtype=torch.bool)
        for k in range(data['n_rna'].shape[0]):
            for i in range(data['n_rna'].shape[1]):
                if cluster2cat[k]=='healthy':
                    temp_rna[0,i] += data['n_rna'][k,i]
                    temp_mask[0,i] = (temp_mask[0,i] or data['masks']['RNA'][k,i])
                    temp_mask_sc[0,i,:] = (temp_mask_sc[0,i,:] | data['masks']['SingleCell'][k,i,:])

                else:
                    temp_rna[1,i] += data['n_rna'][k,i]
                    temp_mask[1,i] = (temp_mask[1,i] or data['masks']['RNA'][k,i])
                    temp_mask_sc[1,i,:] = (temp_mask_sc[1,i,:] | data['masks']['SingleCell'][k,i,:])
            
            
        data['n_rna'] = temp_rna
        data['masks']['RNA'] = temp_mask
        data['masks']['SingleCell'] = temp_mask_sc

        self.cluster2clonelabel = ['healthy','tumor']
        self.cluster2cat = ['healthy','tumor']
        self.clonelabel2cat = ['healthy','tumor']
        self.clonlabel2clusters = {'healthy': [0], 'tumor': [1]}
        self.cat2clusters = {'healthy': [0], 'tumor': [1]}
        self.cat2clonelabels = {'healthy': ['healthy'], 'tumor': ['tumor']}
        self.clonelabel2cat = {'healthy':'healthy', 'tumor':'tumor'}
        self._init_cat_clonelabel()
        return data