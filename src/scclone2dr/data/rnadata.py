from ..utils import *
import torch
import pandas as pd
import numpy as np
import os


class RNAData():
    """
    - For scRNA-seq data, the data should be saved as a dataframe with rows being cells and columns being: dim1, ..., dim "number of total latent dim", clone type (ranging from 0 to Kmax) (optional: the celltype of the cell).
    - 
    
    A separate dataframe should contain as rows the clonelabels and the category of the clone (either healthy or tumor).
    """
    def __init__(self, path_rna):
        self.path_rna = path_rna
        if not(path_rna is None):
            self.dfclones = pd.read_csv(self.path_rna+'clone_infos.csv', index_col=0)
            self.Kmax = self.dfclones.shape[0]
            self.sample_names = [self.get_sampleID_from_file(file) for file in os.listdir(os.path.join(self.path_rna, 'sample2data'))]
            self.sample2rna_files = {self.get_sampleID_from_file(file):file for file in os.listdir(os.path.join(self.path_rna, 'sample2data'))}
            

    def get_sampleID_from_file(self, file):
        idx = 1
        while (idx<len(file) and file[idx]!='-' and file[idx:]!='.csv'):
            idx+=1
        return file[:idx]


    def cloneID2clonetype(self, sample, cloneID):
        return self.dfclones.iloc[cloneID, :]['clonetype_{0}'.format(sample)]

    def cloneID2clonelabel(self, cloneID):
        return self.dfclones.iloc[cloneID, :]['clonelabel']
    
    def cloneID2clonecat(self, cloneID):
        return self.dfclones.iloc[cloneID, :]['clonecategory']
    
#     def clonetype2clonelabel(self, clonetype):
#         cloneID = self.dfclones.index[df['clonetype'] == clonetype].tolist()[0]
#         return self.dfclones.iloc[cloneID, :]['clonelabel']
    
    def clonelabel2clonecat(self, clonelabel):
        cloneID = self.dfclones.index[df['clonelabel'] == clonelabel].tolist()[0]
        return self.dfclones.iloc[cloneID, :]['clonecategory']
    
    def clonelabel2clonecat(self, clonelabel):
        cloneID = self.dfclones.index[df['clonelabel'] == clonelabel].tolist()[0]
        return self.dfclones.iloc[cloneID, :]['clonecategory']
    
    def load_data(self, sample):
        df = pd.read_csv(os.path.join(os.path.join(self.path_rna, 'sample2data'), self.sample2rna_files[sample]), index_col=0)
        return df

    def get_features(self, sample, cloneID=None):
        df = self.load_data(sample)
        if cloneID is None:
            return np.ones(df.shape[0]), df.loc[[col for col in df.columns if 'dim' in col]].to_numpy()
        else:
            return np.where((df['cloneID']==cloneID).values)[0], df.loc[df['cloneID']==cloneID,:].loc[:,[col for col in df.columns if 'dim' in col]].to_numpy()