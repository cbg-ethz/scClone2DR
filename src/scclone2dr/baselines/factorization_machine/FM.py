import torch
import torch.nn as nn
import torch.optim as optim
from ...resultanalysis import *
from ...datasets import *
import pickle
from copy import deepcopy


class FM_model(nn.Module):
    def __init__(self, D=None, dim_drug_feat=10, dim_pathways=39):
        super().__init__()
        self.bias = nn.Parameter(torch.ones(D))
        self.drug_feat = nn.Parameter(torch.ones(D,dim_drug_feat))
        self.W = nn.Parameter(torch.ones(dim_drug_feat, dim_pathways))
        self.normalize = dim_drug_feat * dim_pathways
    
    def forward(self, subclone_features):
        # return the ratio pi_k/pi_0 of shape:  D x (total subclones)
        # subclone_features: dim feat x (Kmax x Nsamples)
        return torch.exp(self.bias.unsqueeze(1) + self.drug_feat @ self.W @ subclone_features / self.normalize)


class FM(BaseDataset, ComputeStatistics):
    def __init__(self, cluster2clonelabel, clonelabel2cat, use_true_proportions=False):
        ComputeStatistics.__init__(self)
        BaseDataset.__init__(self)
        self.cluster2clonelabel = cluster2clonelabel
        self.clonelabel2cat = clonelabel2cat
        self.init_cat_clonelabel()        
        self.use_true_proportions = use_true_proportions

    def eval(self, data, true_params=None):    
        Kmax, N, dim_pathways = data['X'].shape
        N = data['n_r'].shape[2]
        D = data['D']
        frac_r = 1. - data['n0_r'] / data['n_r']
        mean_frac_r = torch.sum(data['masks']['R']*torch.nan_to_num(frac_r), dim=0) / torch.sum(data['masks']['R'], dim=0)
    
        if self.use_true_proportions:
            proportions_rna = data['proportions']
        else:
            ### 2) Correcting RNA proportions using control wells
            proportions_rna = (data['n_rna'] / torch.sum(data['n_rna'], dim=0).unsqueeze(0)).T        
        proportions = torch.zeros((N,Kmax))
        for i in range(N):
            lsk =  [k for k,el in enumerate(data['masks']['C'][:,i]) if el]
            pi0fd = 1.-torch.mean(data['frac_c'][lsk,i])
            proportions[i,0] = pi0fd
            proportions[i,1:] = (1-pi0fd) * proportions_rna[i,1:] / torch.sum(proportions_rna[i,1:])
    
         
        # Vectorize
        vec_proportions = []
        vec_true_proportions = []
        subclone_features = []
        indexes_subclones = []
        for i in range(N):
            for k in range(Kmax):
                vec_proportions.append(proportions[i,k])
                vec_true_proportions.append(data['proportions'][-N:,:][i,k])
                subclone_features.append(list(data['X'][k,i,:]))
                indexes_subclones.append(i)
        subclone_features = torch.tensor(subclone_features).T
        vec_proportions = torch.tensor(vec_proportions)
        vec_true_proportions = torch.tensor(vec_true_proportions)
        indexes_subclones = torch.tensor(indexes_subclones)

        with torch.no_grad():
            self.model.eval()
            sub_ratios_pi = self.model(subclone_features) 
            sub_scores = sub_ratios_pi * vec_proportions
            sub_scores_true_props = sub_ratios_pi * vec_true_proportions
            hat_scores = torch.zeros((D,N))
            hat_scores_true_props = torch.zeros((D,N))
            input = torch.zeros(N)
            for d in range(D):
                hat_scores[d,:] = input.scatter_reduce(0, indexes_subclones, sub_scores[d,:], reduce="sum")
                hat_scores_true_props[d,:] = input.scatter_reduce(0, indexes_subclones, sub_scores_true_props[d,:], reduce="sum")
            ratios_pi = torch.zeros((D,N,Kmax))
            for d in range(D):
                ratios_pi[d,:,:] = torch.reshape(sub_ratios_pi[d,:], (N,Kmax))

        fold_change_pred = torch.log(torch.sum( torch.reshape(ratios_pi, (D, N, Kmax)) * proportions[-N,:].unsqueeze(0), dim=2))
        
        ratios_pi = ratios_pi.permute(0,2,1)
        pi = deepcopy(ratios_pi)
        D, Kmax, N = pi.shape
        for d in range(D):
            for i in range(N):
                pi[d,:,i] *= torch.sum(true_params['pi'][d,:,i]) / torch.sum(pi[d,:,i])
        self.compute_KL_survival_proba(data, {'pi':pi})
        self.compute_error_overall_survival(data, {'pi':pi, 'proportions':proportions})
        self.compute_spearman_drug(data, data, params={'pi':ratios_pi})
        self.compute_drug_effects({'pi':ratios_pi, 'proportions':proportions}, true_params=data)
        self.compute_spearman_subclone(data, data, params={'pi':ratios_pi})
        self.results['fold_change_pred'] = (fold_change_pred.permute(1,0)).reshape(-1).numpy()
        self.results['pi'] = pi


    def train(self, data_train, nb_epochs=1000, lr=0.1, verbose=False):
        Kmax, N, dim_pathways = data_train['X'].shape
        N = data_train['n_r'].shape[2]
        D = data_train['D']
        frac_r = data_train['n0_r'] / data_train['n_r']
        mean_frac_r = torch.sum(data_train['masks']['R']*torch.nan_to_num(frac_r), dim=0) / torch.sum(data_train['masks']['R'], dim=0) 
        frac_c = data_train['n0_c'] / data_train['n_c']
        mean_frac_c = torch.sum(data_train['masks']['C']*torch.nan_to_num(frac_c), dim=0) / torch.sum(data_train['masks']['C'], dim=0)
        scores = (mean_frac_c[:N].unsqueeze(0)) / (mean_frac_r) # D x N


        if self.use_true_proportions:
            proportions_rna = data_train['proportions']
        else:
            ### 2) Correcting RNA proportions using control wells
            proportions_rna = (data_train['n_rna'] / torch.sum(data_train['n_rna'], dim=0).unsqueeze(0)).T
        proportions = torch.zeros((N,Kmax))
        for i in range(N):
            lsk =  [k for k,el in enumerate(data_train['masks']['C'][:,i]) if el]
            pi0fd = 1.-torch.mean(data_train['frac_c'][lsk,i])
            proportions[i,0] = pi0fd
            proportions[i,1:] = (1-pi0fd) * proportions_rna[i,1:] / torch.sum(proportions_rna[i,1:])


        
        # Vectorize
        vec_proportions = []
        subclone_features = []
        indexes_subclones = []
        for i in range(N):
            for k in range(Kmax):
                vec_proportions.append(proportions[i,k])
                subclone_features.append(list(data_train['X'][k,i,:]))
                indexes_subclones.append(i)
        subclone_features = torch.tensor(subclone_features).T
        vec_proportions = torch.tensor(vec_proportions)
        indexes_subclones = torch.tensor(indexes_subclones)



        self.model = FM_model(D=D, dim_pathways=dim_pathways)
        loss_fn = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr, betas=(0.90, 0.999))


        for step in range(nb_epochs):
            # compute loss
            sub_scores = self.model(subclone_features) * vec_proportions.unsqueeze(0) # dim: D x (Kmax x N)
            hat_scores = torch.zeros((D,N))
            input = torch.zeros(N)
            for d in range(D):
                hat_scores[d,:] = input.scatter_reduce(0, indexes_subclones, sub_scores[d,:], reduce="sum")
            loss = loss_fn(hat_scores, scores)        
            loss.backward()
            # take a step and zero the parameter gradients
            optimizer.step()
            optimizer.zero_grad()

            if (step % 100 == 0) and verbose:
                print('[iter {}]  loss: {:.4f}'.format(step, loss))