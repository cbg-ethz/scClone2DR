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
sys.path.append('/data/users/quentin/final_package/')
import scClone2DR as sccdr
import matplotlib.pyplot as plt
import numpy as np
from copy import deepcopy
import pickle
import torch
from tqdm import tqdm
import os
import plotly.io as pio
rootpath = '/data/users/quentin/final_package/experiments/paper_results'
import h5py

gene_set_collections = ['c6','hallmarks','c2_pid','geneOncoKB']
#gene_set_collections = ['geneOncoKB']

cohort2clonemodes = {'melanoma': ['scatrex','phenograph'], 'aml':['phenograph']} #{'melanoma': [ 'scatrex', 'phenograph'], 'aml':['phenograph']}
feat2penl1 = {'geneOncoKB':51.2,'gene':51.2,'c6':3.2,'hallmarks':3.2,'c2_pid':12.8}
feat2penl2 = {'geneOncoKB':3.2,'gene':3.2,'c6':1.6,'hallmarks':3.2,'c2_pid':3.2}
penalty_l2 = 3.2

for gene_set_collection in gene_set_collections:
    for COHORT in ['melanoma','aml']:
        for clonemode in cohort2clonemodes[COHORT]:
            print("Starting",COHORT, gene_set_collection, clonemode)
            
            try:
                penalty_l2 = feat2penl2[gene_set_collection]
                if 'gene' in gene_set_collection:
                    lspenl1 = [12.8, 51.2]
                else:
                    lspenl1 = [feat2penl1[gene_set_collection]]

                for pl1idx, penalty_l1 in enumerate(lspenl1):
                    if pl1idx == 0:
                        addi = ''
                    else:
                        addi = '_sparser'

                    pathsave = os.path.join(rootpath,'{0}/{1}_{2}{3}'.format(COHORT+"_allbaselines", gene_set_collection, clonemode, addi))
                    if not os.path.exists(pathsave):
                        os.makedirs(pathsave)

                    mode_features = 'metacells_{0}_{1}'.format(gene_set_collection, clonemode)

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

                    idxs_train = [i for i in range(int(data_ref['N']))]
                    idxs_test = []

                    data_train, data_test, sample_names_train, sample_names_test = model.get_real_data_split(idxs_train, idxs_test)

                    if False:
                        ## FM
                        modelFM = sccdr.models.FM_sc(model.cluster2clonelabel, model.clonelabel2cat)
                        modelFM.train(data_train)
                        modelFM.eval(data_train)
                        li_weights = modelFM.get_local_importance_weights(data_train)
                        li_weights = li_weights.detach().numpy()
                        latent_dim = li_weights.shape[-1]
                        for j in range(latent_dim):
                            for d in range(model.D):
                                li_weights[d,:,:,j][~(data_train['masks']['RNA'])] = float('nan')
                        all_local_importances = li_weights
                        with h5py.File(os.path.join(pathsave,'factorization_machine_local_importance.h5'), 'w') as f:
                            # Create a dataset
                            dset = f.create_dataset('local_importance_mean', data=all_local_importances)

                            column_names = {
                            'dim2_subclones': [i for i in range(data_train['Kmax'])],
                            'dim3_samples': sample_names_train, #np.char.encode(sample_names_test, 'ascii'),
                            'dim4_dimensions': model.feature_names,
                            'dim1_drugs':  model.FD.selected_drugs
                            }

                            for dim, labels in column_names.items():
                                dset.attrs[dim] = labels

                        postmean = modelFM.pi.numpy()
                        for d in range(data_train['D']):
                            postmean[d,:,:][~(data_train['masks']['RNA'])] = float('nan')
                        with h5py.File(os.path.join(pathsave,'factorization_machine_survival_probabilities.h5'), 'w') as f:
                            # Create a dataset
                            dset = f.create_dataset('survival_probabilities', data=postmean)
                            column_names = {
                            'dim2_subclones': [i for i in range(data_train['Kmax'])],
                            'dim3_samples': sample_names_train, # np.char.encode(sample_names, 'ascii'),
                            'dim1_drugs': model.FD.selected_drugs
                            }
                            for dim, labels in column_names.items():
                                dset.attrs[dim] = labels

                    ## BASE
                    if False:
                        model_base = sccdr.models.scClone2DR(path_fastdrug=path_fastdrug, path_rna=path_rna, type_guide="lowrank_MVN", rank=10)
                        data_base_ref = model_base.get_real_data(concentration_DMSO=concentration_DMSO, concentration_drug=concentration_drug)
                        del data_base_ref
                        model_base.data['X'] = torch.zeros(model_base.data['X'].shape)
                        data_base_train, _, sample_names_train, sample_names_test = model_base.get_real_data_split(idxs_train, idxs_test)
                        params_svi = model_base.train(data_base_train, penalty_l1=penalty_l1, penalty_l2=penalty_l2, lr=0.005, n_steps=2000)
                        with open(os.path.join(rootpath,'{0}/{1}_{2}{3}/{4}params_svi.pkl'.format(COHORT+"_allbaselines", gene_set_collection, clonemode, addi, 'base_')), 'wb') as handle:
                            pickle.dump(params_svi, handle, protocol=pickle.HIGHEST_PROTOCOL)
                        posterior_mean_params = model_base.sampling_from_posterior(data_base_train, pathsave, params=params_svi, nb_ites=30, sample_names=model_bimodal.sample_names, model_name='bimodal_')
                        with open(os.path.join(rootpath,'{0}/{1}_{2}{3}/{4}posterior_mean_params.pkl'.format(COHORT+"_allbaselines", gene_set_collection, clonemode, addi, 'bimodal_')), 'wb') as handle:
                            pickle.dump(posterior_mean_params, handle, protocol=pickle.HIGHEST_PROTOCOL)



                    ## BULK                
                    model_bulk = sccdr.models.scClone2DR()
                    databulk = model_bulk.get_bulk_from_real_data(data_train, model.cluster2cat, model.cat2clusters)
                    model_bulk.data = databulk
                    model_bulk.D = model.D
                    model_bulk.R = model.R
                    model_bulk.C = model.C
                    model_bulk.N = model.N
                    model_bulk.latent_dim = model.latent_dim
                    model_bulk.feature_names = model.feature_names
                    model_bulk.sample_names = model.sample_names
                    model_bulk.rank = model.rank
                    model_bulk.FD = model.FD
                    databulk, _ = model_bulk.add_design_preassay_bulk(databulk, {}, model_bulk.sample_names, [])
                    model_bulk.data = databulk
                    data_bulk_train, _, sample_names_train, sample_names_test = model_bulk.get_real_data_split(idxs_train, idxs_test)
                    params_svi = model_bulk.train(data_bulk_train, penalty_l1=penalty_l1, penalty_l2=penalty_l2, lr=0.005,  n_steps=1000)
                    with open(os.path.join(rootpath,'{0}/{1}_{2}{3}/{4}params_svi.pkl'.format(COHORT+"_allbaselines", gene_set_collection, clonemode, addi, 'bulk_')), 'wb') as handle:
                        pickle.dump(params_svi, handle, protocol=pickle.HIGHEST_PROTOCOL)
                    posterior_mean_params = model_bulk.sampling_from_posterior(data_bulk_train, pathsave, params=params_svi, nb_ites=200, sample_names=model_bulk.sample_names, model_name='bulk_')
                    with open(os.path.join(rootpath,'{0}/{1}_{2}{3}/{4}posterior_mean_params.pkl'.format(COHORT+"_allbaselines", gene_set_collection, clonemode, addi, 'bulk_')), 'wb') as handle:
                        pickle.dump(posterior_mean_params, handle, protocol=pickle.HIGHEST_PROTOCOL)
                    del posterior_mean_params
                    del params_svi
                    del data_bulk_train
                    del databulk


                    ## BIMODAL
                    model_bimodal = sccdr.models.scClone2DR()
                    databimodal = model_bimodal.get_bimodal_from_real_data(data_train, model.cluster2cat, model.cat2clusters)
                    model_bimodal.data = databimodal
                    model_bimodal.D = model.D
                    model_bimodal.R = model.R
                    model_bimodal.C = model.C
                    model_bimodal.N = model.N
                    model_bimodal.latent_dim = model.latent_dim
                    model_bimodal.feature_names = model.feature_names
                    model_bimodal.sample_names = model.sample_names
                    model_bimodal.rank = model.rank
                    model_bimodal.FD = model.FD
                    databimodal, _ = model_bimodal.add_design_preassay(databimodal, {}, model_bimodal.sample_names, [])
                    model_bimodal.data = databimodal
                    data_bimodal_train, _, sample_names_train, sample_names_test = model_bimodal.get_real_data_split(idxs_train, idxs_test)
                    params_svi = model_bimodal.train(data_bimodal_train, penalty_l1=penalty_l1, penalty_l2=penalty_l2, lr=0.005, n_steps=1000)
                    with open(os.path.join(rootpath,'{0}/{1}_{2}{3}/{4}params_svi.pkl'.format(COHORT+"_allbaselines", gene_set_collection, clonemode, addi, 'bimodal_')), 'wb') as handle:
                        pickle.dump(params_svi, handle, protocol=pickle.HIGHEST_PROTOCOL)
                    posterior_mean_params = model_bimodal.sampling_from_posterior(data_bimodal_train, pathsave, params=params_svi, nb_ites=200, sample_names=model_bimodal.sample_names, model_name='bimodal_')
                    with open(os.path.join(rootpath,'{0}/{1}_{2}{3}/{4}posterior_mean_params.pkl'.format(COHORT+"_allbaselines", gene_set_collection, clonemode, addi, 'bimodal_')), 'wb') as handle:
                        pickle.dump(posterior_mean_params, handle, protocol=pickle.HIGHEST_PROTOCOL)
                    del posterior_mean_params
                    del params_svi
                    del data_bimodal_train
                    del databimodal


                    del data_train
                    del data_ref
            except:
                pass
