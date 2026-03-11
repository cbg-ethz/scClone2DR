import sys
sys.path.append('../../')
import scClone2DR as sccdr
import matplotlib.pyplot as plt
import numpy as np
from copy import deepcopy
import pickle
import torch
from tqdm import tqdm
import plotly.io as pio


import sys
sys.path.append('/data/users/quentin/package_paper/')
import scClone2DR as sccdr
import matplotlib.pyplot as plt
import numpy as np
from copy import deepcopy
import pickle
import torch
from tqdm import tqdm
import os
import random
import plotly.io as pio
rootpath = '/data/users/quentin/package_paper/experiments/paper_results'
np.float_ = np.float64

gene_set_collections = ['geneOncoKB'] #['c6','hallmarks', 'c2_pid', 'c2_kegg_medicus']
cohort2clonemodes = {'melanoma': ['scatrex'], 'aml':['phenograph']}


penalties_l1 = [0.2*2**i for i in range(3,10)]
penalties_l2 = [0.05*2**i for i in range(3,10)]




stabilityname = 'stability_paper'

for COHORT in ['melanoma']:
    for clonemode in cohort2clonemodes[COHORT]:
        for gene_set_collection in gene_set_collections:
            if gene_set_collection=='gene':
                penalties_l1 = [51.2]
                penalties_l2 = [3.2]
            elif gene_set_collection=="geneOncoKB":
                penalties_l1 = [1.]
                penalties_l2 = [1.]
            else:
                penalties_l1 = [3.2]
                penalties_l2 = [3.2]
            for pl1, penalty_l1 in tqdm(enumerate(penalties_l1)):
                for penalty_l2 in penalties_l2:
            
                    tempdir = os.path.join(rootpath,'{0}/{1}_{2}_rawcounts_scvi/{3}/'.format(COHORT, gene_set_collection, clonemode, stabilityname))
                    if not os.path.exists(tempdir):
                        os.makedirs(tempdir)

                    mode_features = 'metacells_{0}_{1}_rawcounts_scvi'.format(gene_set_collection, clonemode)

                    directory = os.path.join(rootpath,COHORT)
                    if not os.path.exists(directory):
                        os.makedirs(directory)
                    pathsave = os.path.join(rootpath,COHORT,gene_set_collection+"_rawcounts_scvi")
                    if not os.path.exists(pathsave):
                        os.makedirs(pathsave)

                    if COHORT=='melanoma':
                        path_rna = '/data/users/04_share_reanalysis_results/melanoma_2025/02_atypical_removed_preprocessing/{0}/'.format(mode_features)
                        path_fastdrug = '/data/users/04_share_reanalysis_results/melanoma_2025/MEL_CNN_abundances_no_plate_effect_correction.csv'
                        concentration_DMSO = '100'
                        concentration_drug = '5'
                    elif COHORT=='aml':
                        concentration_DMSO = '200'
                        concentration_drug = '10'
                        path_rna = '/data/users/04_share_reanalysis_results/aml_2025/02_atypical_removed_preprocessing/{0}/'.format(mode_features)
                        path_fastdrug = '/data/users/04_share_reanalysis_results/01_aml/AML_PCY_cell_numbers_no_plate_effect_correction.csv'
                        path_info_cohort = '/data/users/04_share_reanalysis_results/01_aml/2024-08-15_aml_overview_scRNA.tsv'

                    model = sccdr.models.scClone2DR(path_fastdrug=path_fastdrug, path_rna=path_rna, type_guide="lowrank_MVN", rank=10)
                    data_ref = model.get_real_data(concentration_DMSO=concentration_DMSO, concentration_drug=concentration_drug)

                    for ite in range(0,100):
                        try:
                            if clonemode=='scatrex' and gene_set_collection=='gene':
                                np.random.seed(ite)
                                idxs = np.arange(int(data_ref['N']))
                                np.random.shuffle(idxs)  # NumPy's shuffle respects the seed
                                idxs_train = idxs[:int(0.63*data_ref['N'])]
                                idxs_test = idxs[int(0.63*data_ref['N']):]
                            else:
                                # this is to make sure we use the same data split accross the different settings
                                #with open(os.path.join(rootpath,'{0}/{1}_{2}/stability_paper/params_svi_ite_{3}_l1_12.8_l2_3.2.pkl'.format(COHORT, 'gene', 'scatrex', ite)), 'rb') as handle:
                                with open(os.path.join('/data/users/quentin/final_package/experiments/paper_results','{0}/{1}_{2}/stability_paper/params_svi_ite_{3}_l1_12.8_l2_3.2.pkl'.format(COHORT, 'gene', 'scatrex', ite)), 'rb') as handle:

                                    params_scatrex = pickle.load(handle)
                                    samples_names_train = params_scatrex['sample_names_train']
                                    samples_names_test = params_scatrex['sample_names_test']
                                    idxs_train = []
                                    idxs_test = []
                                    for sample in samples_names_train:
                                        idx_s = np.where(model.sample_names==sample)[0][0]
                                        idxs_train.append(idx_s)
                                    for sample in samples_names_test:
                                        idx_s = np.where(model.sample_names==sample)[0][0]
                                        idxs_test.append(idx_s)

                            data_train, data_test, sample_names_train, sample_names_test = model.get_real_data_split(idxs_train, idxs_test)
                            if gene_set_collection=='gene':
                                params_svi = model.train(data_train, penalty_l1=penalty_l1, penalty_l2=penalty_l2, n_steps=2000)
                            else:
                                params_svi = model.train(data_train, penalty_l1=penalty_l1, penalty_l2=penalty_l2 , n_steps=2000)
                            params_svi['sample_names_train'] = sample_names_train
                            params_svi['sample_names_test'] = sample_names_test
                            params_svi['idxs_train'] = idxs_train
                            params_svi['idxs_test'] = idxs_test

                            with open(os.path.join(rootpath,'{0}/{1}_{2}_rawcounts_scvi/{6}/params_svi_ite_{5}_l1_{3}_l2_{4}.pkl'.format(COHORT, gene_set_collection, clonemode, str(penalty_l1), str(penalty_l2), ite, stabilityname)), 'wb') as handle:
                                pickle.dump(params_svi, handle, protocol=pickle.HIGHEST_PROTOCOL)

                        except:
                            pass
