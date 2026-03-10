from ..utils import *
import torch
import pandas as pd
import numpy as np

class FastDrug():
    def __init__(self, path_fastdrug):
        if path_fastdrug is None:
            pass
        else:
            self.path_fastdrug = path_fastdrug
            df = pd.read_csv(self.path_fastdrug)
            self.sample_names = df['SampleID'].unique()
    
    def load_FD_data(self, sample_names, concentration_DMSO='100', concentration_drug='10'):
        # We select drug for which we have measurement for the specified concentration  (and we keep all patients: we will mask if no data for a pair).
        df = pd.read_csv(self.path_fastdrug)
        self.sample_names = sample_names
        df = df.loc[df['Concentration']==concentration_drug]
        df = df.loc[df['SampleID'].apply(lambda x: x in sample_names)]
        setdrugs = df['Drug'].unique()

        setdrugs = np.delete(setdrugs,  np.where(setdrugs == 'DMSO'))
        # removing drug combinations
        selected_drugs = [drug for drug in setdrugs if not('|' in drug)]
        print('Number of drugs', len(selected_drugs))



        ########## LOAD DRUG DATA: Number of cells for the wells
        sample2nb_cells_per_well = {}
        sample2tumor = {}
        sample2wellpos = {}

        for drug in selected_drugs:
            for sample in sample_names:
                lsMEL = df[(df['SampleID']==sample) & (df['Concentration']==concentration_drug) & (df['Drug']==drug)]['Number_tumor_cells']
                lsMEL_clean = [x for x in lsMEL if str(x) != 'nan']

                # well positions
                lsMEL_wellpos = df[(df['SampleID']==sample) & (df['Concentration']==concentration_drug) & (df['Drug']==drug)][['Well_position_1','Well_position_2']]
                lsMEL_wellpos = lsMEL_wellpos.to_numpy()
                valid_idxs = [i for i,x in enumerate(lsMEL) if str(x) != 'nan']
                lsMEL_wellpos = lsMEL_wellpos[valid_idxs,:]

                sample2tumor[(drug,sample)] = lsMEL_clean
                sample2wellpos[(drug,sample)] = lsMEL_wellpos

                ls = df[(df['SampleID']==sample) & (df['Concentration']==concentration_drug) & (df['Drug']==drug)]['Number_all_cells']
                ls_clean = [x for x in ls if str(x) != 'nan']
                sample2nb_cells_per_well[(drug,sample)] = ls_clean

        df = pd.read_csv(self.path_fastdrug)
        df = df.loc[df['SampleID'].apply(lambda x: x in sample_names)]
        dfcontrol = df[(df['Drug']=='DMSO') & (df['Concentration']==concentration_DMSO)][['SampleID', 'Number_tumor_cells', 'Well_position_1', 'Well_position_2']].dropna()
        sample2tumor_control = dfcontrol.groupby('SampleID')['Number_tumor_cells'].apply(list).to_dict()

        # get wells
        sample2control_wellpos = {}
        well1 = dfcontrol.groupby('SampleID')['Well_position_1'].apply(list).to_dict()
        well2 = dfcontrol.groupby('SampleID')['Well_position_2'].apply(list).to_dict()
        for sample in sample_names:
            sample2control_wellpos[sample] = np.concatenate((np.array(well1[sample]).reshape(-1,1),np.array(well2[sample]).reshape(-1,1)), axis=1)



        df = pd.read_csv(self.path_fastdrug)
        df = df.loc[df['SampleID'].apply(lambda x: x in sample_names)]
        dfcontrol = df[(df['Drug']=='DMSO') & (df['Concentration']==concentration_DMSO)][['SampleID','Number_all_cells']].dropna()
        sample2nb_cells_per_well_control = dfcontrol.groupby('SampleID')['Number_all_cells'].apply(list).to_dict()

        print('Number of samples: ', len(sample_names))
       
        self.selected_drugs, self.sample2nb_cells_per_well, self.sample2nb_cells_per_well_control, self.sample2tumor, self.sample2tumor_control, self.sample2wellpos, self.sample2control_wellpos = selected_drugs, sample2nb_cells_per_well, sample2nb_cells_per_well_control, sample2tumor, sample2tumor_control, sample2wellpos, sample2control_wellpos
        
        
        

    
        
        