import torch
import torch.nn as nn
import torch.optim as optim
from ...resultanalysis import *
from ...datasets import *
import pickle
from copy import deepcopy

class NN_model(nn.Module):
    def __init__(self, dim_feat=1):
        super().__init__()
        self.fc1 = nn.Linear(dim_feat, 128)  # Input size is 28x28 for MNIST dataset
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)
        #self.W = nn.Parameter(props_ini) 
        #self.Ndrug, self.Kmax = props_ini.shape
        #self.D = D
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        x = torch.exp(x)
        return x


class NN(BaseDataset, ComputeStatistics):
    """
    Neural Network model to estimate survival probabilities.
    This class should be used only with simulated data, with feature defined at the subclonal level.
    """
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
        dim_feat = 2*data['X'].shape[2] 
        halfdim_feat = data['X'].shape[2] 
        D = data['n_r'].shape[1]
        Ndrug = data['n_r'].shape[2]
        Kmax = data['X'].shape[0]
        subclone_features = torch.zeros((D,Ndrug,Kmax,dim_feat)) 
        subclone_features[:,:,:,:halfdim_feat] = (data['X'].permute(1,0,2)).unsqueeze(0)
        subclone_features[:,:,:,halfdim_feat:] = (true_params['beta']).unsqueeze(1).unsqueeze(1)
    
        with torch.no_grad():
            self.model.eval()
            ratio_pis = self.model(subclone_features.reshape(D*Ndrug*Kmax,dim_feat))
            fold_change_pred = torch.log(torch.sum( torch.reshape(ratio_pis, (D, Ndrug, Kmax)) * proportions[-Ndrug,:].unsqueeze(0), dim=2))

        pi = deepcopy(ratio_pis)
        pi = torch.reshape(pi, (D,Ndrug,Kmax)).permute(0,2,1)
        for d in range(D):
            for i in range(Ndrug):
                pi[d,:,i] *= torch.sum(true_params['pi'][d,:,i]) / torch.sum(pi[d,:,i])
        self.compute_KL_survival_proba(data, {'pi':pi})
        self.compute_error_overall_survival(data, {'pi':pi, 'proportions':proportions})
        self.compute_spearman_drug(data, data, params={'pi':pi})
        self.compute_drug_effects({'pi':pi, 'proportions':proportions}, true_params=data)
        self.compute_spearman_subclone(data, data, params={'pi':pi})
        self.results['fold_change_pred'] = (fold_change_pred.permute(1,0)).reshape(-1).numpy()
        self.results['pi'] = pi

    def train(self, data_train, beta, nb_epochs=1000, lr=0.01, verbose=False):
        Kmax, N, dim_pathways = data_train['X'].shape
        N = data_train['n_r'].shape[2]
        D = data_train['D']
        frac_r = data_train['n0_r'] / data_train['n_r']
        mean_frac_r = torch.sum(data_train['masks']['R']*torch.nan_to_num(frac_r), dim=0) / torch.sum(data_train['masks']['R'], dim=0) 
        frac_c = data_train['n0_c'] / data_train['n_c']
        mean_frac_c = torch.sum(data_train['masks']['C']*torch.nan_to_num(frac_c), dim=0) / torch.sum(data_train['masks']['C'], dim=0)
        scores = mean_frac_c[:N].unsqueeze(0) / mean_frac_r # D x N

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
        dim_feat = 2*data_train['X'].shape[2] 
        halfdim_feat = data_train['X'].shape[2] 
        D = data_train['n_r'].shape[1]
        Ndrug = data_train['n_r'].shape[2]
        Kmax = data_train['X'].shape[0]
        subclone_features = torch.zeros((D,Ndrug,Kmax,dim_feat)) 
        vec_proportions = proportions.unsqueeze(0)
        subclone_features[:,:,:,:halfdim_feat] = (data_train['X'].permute(1,0,2)).unsqueeze(0)
        subclone_features[:,:,:,halfdim_feat:] = (beta).unsqueeze(1).unsqueeze(1)
        
        self.model = NN_model(dim_feat=dim_feat) #, props_ini = proportions, D=D)
        loss_fn = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr, betas=(0.90, 0.999))


        for step in range(nb_epochs):
            # compute loss
            sub_scores = self.model(subclone_features.reshape(D*Ndrug*Kmax,dim_feat))
            hat_scores = torch.sum( torch.reshape(sub_scores, (D, Ndrug, Kmax)) * vec_proportions, dim=2)
            loss = loss_fn(hat_scores, scores)        
            loss.backward()
            # take a step and zero the parameter gradients
            optimizer.step()
            optimizer.zero_grad()

            if (step % 100 == 0) and verbose:
                print('[iter {}]  loss: {:.4f}'.format(step, loss))