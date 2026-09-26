import scclone2dr
from scclone2dr.pipeline import scClone2DRPipeline
from scclone2dr.trainer import Trainer, GuideType
import matplotlib.pyplot as plt
import torch
import os
n_steps = 2000


setting = "easy"
if setting=="easy":
    R = 20
    neg_bin = 100
elif setting=="very_veasy":
    R = 30
    neg_bin = 10000
else:
    R = 5
    neg_bin = 2

import sys
sys.path.append('/mydata/harat/quentin/scClone2DR')

def main(response_misspec, noise_misspec, dispersion=50.0, swap_frac=0.):
    from scclone2dr.data.misspecified_simulateddata_sc import ResponseMisspec, NoiseMisspec
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt



    cwd = os.getcwd()
    OUT_DIR = os.path.join(cwd, "results_misspecified/response_misspec_{}_noise_misspec_{}_dispersion_{}_frac_{}/".format(
        response_misspec, noise_misspec, dispersion, swap_frac
    ))
    os.makedirs(OUT_DIR, exist_ok=True)

    from scclone2dr.data.misspecified_simulateddata_sc import ResponseMisspec, NoiseMisspec

    response2method = {
        "None": ResponseMisspec.NONE,
        "Hill": ResponseMisspec.HILL,
        "Saturating": ResponseMisspec.SATURATING,
        "Threshold": ResponseMisspec.THRESHOLD,
    }
    noise2method = {
        "None": NoiseMisspec.NONE,
        "Heteroscedastic": NoiseMisspec.HETEROSCEDASTIC,
        "HeavyTail": NoiseMisspec.HEAVY_TAIL,
        "ZeroInflated": NoiseMisspec.ZERO_INFLATED,
    }
    response_misspec = response2method[response_misspec]
    noise_misspec = noise2method[noise_misspec]
    datamodule_misspec = scclone2dr.data.MisspecifiedSimulatedData(response_misspec=response_misspec,
        noise_misspec=noise_misspec,
        noise_dispersion=dispersion,
        swap_frac=swap_frac
    )

    DATA_DIR = os.path.join(OUT_DIR, "data/")
    os.makedirs(DATA_DIR, exist_ok=True)
    data_ref, gt_params = datamodule_misspec.generate_and_save_misspecified_data({'C':24,'R':R,'N':60,'Kmax':7, 'D':30, 'theta_rna':15}, neg_bin_n=neg_bin, mode_nu="noise_correction", mode_theta="not shared decoupled", save_dir=DATA_DIR)


    path_rna = DATA_DIR
    path_fastdrug = os.path.join(DATA_DIR, "FD_data.csv")
    datamodule = scclone2dr.data.RealData(path_fastdrug=path_fastdrug, path_rna=path_rna)
    _ = datamodule.get_real_data(concentration_DMSO=5, concentration_drug=5)

    idxs_train = [i for i in range(int(0.5*data_ref['N']))]
    idxs_test = [i for i in range(int(0.5*data_ref['N']), data_ref['N'])]

    data_train, data_test, sample_names_train, sample_names_test = datamodule.get_real_data_split(idxs_train, idxs_test)

    data_train_misspec, data_test_misspec = datamodule_misspec.get_data_split(data_ref, idxs_train, idxs_test)
    gt_params_train, gt_params_test = datamodule_misspec.get_params_split(gt_params, idxs_train, idxs_test)

    if setting not in ["easy", "very_easy"]:
        penalty_l1=0.1
        penalty_l2=0.1
    else:
        penalty_l1=0.02
        penalty_l2=0.02



    # 1) Build the pipeline from the already-loaded real-data module.
    trainer = Trainer(guide_type=GuideType.LOWRANK_MVN, rank=10)
    pipeline = scClone2DRPipeline(
        data_source=datamodule,
        trainer=trainer,
        mode_nu="noise_correction",
        mode_theta="not shared decoupled",
    )

    # Ensure model topology is configured from the dataset metadata.
    pipeline.model.configure(datamodule)

    # 2) Fit on the already prepared data_train dictionary.
    params_svi = pipeline.fit(
        data=data_train,
        penalty_l1=penalty_l1,
        penalty_l2=penalty_l2,
        lr=0.01,
        n_steps=2000,
    )

    posterior_results = pipeline.sample_posterior(
        data=data_test,
        idxs_sample_eval=idxs_test,
        nb_ites=100,
        dir_save=None,
        sample_names=idxs_test,
        model_name="simu_data_",
    )
    evaluations = pipeline.evaluate(data_test, posterior_results['params'], true_params=gt_params_test)


    ################################# BULK

    databulk, datamodule_bulk = datamodule_misspec.get_bulk_from_data(data_ref)
    data_train_bulk, data_test_bulk = datamodule_misspec.get_data_split(databulk, idxs_train, idxs_test)

    # 1) Build the pipeline from the already-loaded real-data module.
    trainer_bulk = Trainer(guide_type=GuideType.NONE)
    pipeline_bulk = scClone2DRPipeline(
        data_source=datamodule_bulk,
        trainer=trainer_bulk,
        mode_nu="noise_correction",
        mode_theta="not shared decoupled",
    )

    # Ensure model topology is configured from the dataset metadata.
    pipeline_bulk.model.configure(datamodule_bulk)

    # 2) Fit on the already prepared data_train dictionary.
    params_svi_bulk = pipeline_bulk.fit(
        data=data_train_bulk,
        penalty_l1=penalty_l1,
        penalty_l2=penalty_l2,
        lr=0.01,
        n_steps=2000,
    )

    posterior_results_bulk = pipeline_bulk.sample_posterior(
        data=data_test_bulk,
        idxs_sample_eval=idxs_test,
        nb_ites=100,
        dir_save=None,
        sample_names=idxs_test,
        model_name="simu_data_",
    )
    from copy import copy
    gt_params_test_bulk = copy(gt_params_test)
    gt_params_test_bulk['proportions'] = data_test_bulk['proportions']
    D,_,N = gt_params_test_bulk['pi'].shape
    pi_test = torch.zeros((D,2,N))
    pi_test[:,0,:] = gt_params_test_bulk['pi'][:,0,:]
    weights = gt_params_test['proportions'][:,1:].clone()
    weights /= torch.sum(weights, dim=1, keepdim=True)
    pi_test[:,1,:] = torch.sum(gt_params_test_bulk['pi'][:,1:,:] * weights.T[None,...], dim=1)
    gt_params_test_bulk['pi'] = pi_test
    evaluations_bulk = pipeline_bulk.evaluate(data_test_bulk, posterior_results_bulk['params'], true_params=gt_params_test_bulk)

    ###################################### BIMODAL

    databimodal, datamodule_bimodal = datamodule_misspec.get_bimodal_from_data(data_ref)
    data_train_bimodal, data_test_bimodal = datamodule_bimodal.get_data_split(databimodal, idxs_train, idxs_test)

    # 1) Build the pipeline from the already-loaded real-data module.
    trainer_bimodal = Trainer(guide_type=GuideType.NONE)
    pipeline_bimodal = scClone2DRPipeline(
        data_source=datamodule_bimodal,
        trainer=trainer_bimodal,
        mode_nu="noise_correction",
        mode_theta="not shared decoupled",
    )

    # Ensure model topology is configured from the dataset metadata.
    pipeline_bimodal.model.configure(datamodule_bimodal)

    # 2) Fit on the already prepared data_train dictionary.
    params_svi_bimodal = pipeline_bimodal.fit(
        data=data_train_bimodal,
        penalty_l1=penalty_l1,
        penalty_l2=penalty_l2,
        lr=0.01,
        n_steps=2000,
    )

    posterior_results_bimodal = pipeline_bimodal.sample_posterior(
        data=data_test_bimodal,
        idxs_sample_eval=idxs_test,
        nb_ites=100,
        dir_save=None,
        sample_names=idxs_test,
        model_name="simu_data_",
    )
    from copy import copy
    gt_params_test_bimodal = copy(gt_params_test)
    gt_params_test_bimodal['proportions'] = data_test_bimodal['proportions']
    D,_,N = gt_params_test_bimodal['pi'].shape
    pi_test = torch.zeros((D,2,N))
    pi_test[:,0,:] = gt_params_test_bimodal['pi'][:,0,:]
    weights = gt_params_test['proportions'][:,1:].clone()
    weights /= torch.sum(weights, dim=1, keepdim=True)
    pi_test[:,1,:] = torch.sum(gt_params_test_bimodal['pi'][:,1:,:] * weights.T[None,...], dim=1)
    gt_params_test_bimodal['pi'] = pi_test
    evaluations_bimodal = pipeline_bimodal.evaluate(data_test_bimodal, posterior_results_bimodal['params'], true_params=gt_params_test_bimodal)


    ################################## BASE

    database, datamodule_base = datamodule_misspec.get_base_from_data(data_ref)
    data_train_base, data_test_base = datamodule_base.get_data_split(database, idxs_train, idxs_test)

    # 1) Build the pipeline from the already-loaded real-data module.
    trainer_base = Trainer(guide_type=GuideType.NONE)
    pipeline_base = scClone2DRPipeline(
        data_source=datamodule_base,
        trainer=trainer_base,
        mode_nu="noise_correction",
        mode_theta="not shared decoupled",
    )

    # Ensure model topology is configured from the dataset metadata.
    pipeline_base.model.configure(datamodule_base)

    # 2) Fit on the already prepared data_train dictionary.
    params_svi_base = pipeline_base.fit(
        data=data_train_base,
        penalty_l1=penalty_l1,
        penalty_l2=penalty_l2,
        lr=0.01,
        n_steps=2000,
    )

    posterior_results_base = pipeline_base.sample_posterior(
        data=data_test_base,
        idxs_sample_eval=idxs_test,
        nb_ites=100,
        dir_save=None,
        sample_names=idxs_test,
        model_name="simu_data_",
    )
    from copy import copy
    gt_params_test_base = copy(gt_params_test)
    gt_params_test_base['proportions'] = data_test_base['proportions']
    evaluations_base = pipeline_base.evaluate(data_test_base, posterior_results_base['params'], true_params=gt_params_test_base)


    ##################### FM
    Kmax, N, n_cells, latent_dim = data_train['X'].shape
    N = data_train['n_r'].shape[2]
    D = data_train['D']
    modelFM = scclone2dr.baselines.FM_sc(datamodule.cluster2clonelabel, datamodule.clonelabel2cat)
    modelFM.train(data_train)
    modelFM.eval(data_test, true_params = gt_params_test)


    #################### NN
    Kmax, N, n_cells, latent_dim = data_train['X'].shape
    N = data_train['n_r'].shape[2]
    D = data_train['D']
    modelNN = scclone2dr.baselines.NN_sc(datamodule.cluster2clonelabel, datamodule.clonelabel2cat)
    modelNN.train(data_train)
    modelNN.eval(data_test, true_params = gt_params_test)



    ############################ RES 
    res = {}
    res['us'] = evaluations
    res['base'] = evaluations_base
    res['bulk'] = evaluations_bulk
    res['bimodal'] = evaluations_bimodal
    res['fm'] = modelFM.r
    res['nn'] = modelNN.r

    # save as pickle
    import pickle
    with open(os.path.join(OUT_DIR, "results.pkl"), "wb") as f:
        pickle.dump(res, f)
    
    import seaborn as sns
    colors_models = sns.color_palette('Set2')

    #models = ['us', 'base', 'bimodal','bulk','fm','fm_true_props','nn','nn_true_props']
    models = ['us','fm','nn','bimodal','bulk', 'base']
    model2name = {'us':'scClone2DR', 'base':'Baseline', 'bimodal':'Dual bulk','bulk':'Bulk','fm':'FM','nn':'NN', 
                'fm_true_props': 'FM true props','nn_true_props':'NN true props'}


    for m in models:
        x = np.abs(1.-res[m].true_drug_effects.numpy())
        argidxs = np.argsort(x)
        argidxs_dec = np.flip(argidxs, axis=0)
        x = x[argidxs_dec]
        err = np.cumsum(np.abs(res[m].true_drug_effects.numpy()-res[m].drug_effects.numpy())[argidxs_dec])/np.cumsum(np.ones(len(x)))
        plt.plot(x, (err), label=model2name[m])
    plt.legend(loc=2)
    plt.ylabel('$L^1$ error on the drug effects', fontsize=14)
    plt.xlabel('Threshold', fontsize=14)
    plt.savefig(os.path.join(OUT_DIR, "drug_effects_error.png"), bbox_inches='tight', dpi=300)
    plt.close()


    import numpy as np
    import seaborn as sns
    import matplotlib.pyplot as plt
    import pandas as pd
    from matplotlib import collections

    # https://stackoverflow.com/questions/70442958/seaborn-how-to-apply-custom-color-to-each-seaborn-violinplot

    # Initialize the dictionary
    dic = {'method': [], "drug scores": [], "statistic": [], 'ground_truth':[]}
    mses = {}

    # Loop through models and calculate MSE
    for m in models:
        estimated_scores = np.array(res[m].drug_scores)
        true_scores = np.array(res[m].true_drug_scores)
        
        # Calculate MSE
        mse = np.mean((estimated_scores - true_scores) ** 2)
        mses[model2name[m]] = mse
        
        # Populate dictionary
        dic['method'] += [model2name[m] for i in range(2 * len(estimated_scores))]
        dic['drug scores'] += list(estimated_scores)
        dic['drug scores'] += list(true_scores)
        dic['ground_truth'] += [False for i in range(len(estimated_scores))]
        dic['ground_truth'] += [True for i in range(len(true_scores))]
        dic['statistic'] += ['estimated ' for i in range(len(estimated_scores))]
        dic['statistic'] += ['ground truth' for i in range(len(true_scores))]

    # Convert to DataFrame
    df = pd.DataFrame(dic)

    # Set theme and style
    sns.set_theme(style='white')
    sns.set_style("ticks", {"xtick.major.size": 12, "ytick.major.size": 12})

    # Create figure with two subplots
    fig, (ax_violin, ax_bar) = plt.subplots(nrows=2, ncols=1, figsize=(10, 8), gridspec_kw={'height_ratios': [3, 1]})

    # Violin plot on top
    colors = {m:colors_models[i] for i,m in enumerate((models))}
    models = ['us','fm','nn','bimodal','bulk', 'base']
    colors = {}
    colors['us'] = "#009E73"
    colors['fm'] = "#0072B2"
    colors['nn'] = "#56B4E9"
    colors['bimodal'] = "#D55E00"
    colors['bulk'] = "#E69F00"
    colors['base'] = "#F0E442"


    ax = sns.violinplot(data=df, x="method", y="drug scores", hue="statistic", split=True, inner="quart", density_norm="width", legend = False, ax=ax_violin)
    for ind, violin in enumerate(ax.findobj(collections.PolyCollection)):
        rgb = colors[models[ind//2]]
        if ind % 2 != 0:
            rgb = "gray"  # make white
        violin.set_facecolor(rgb)
    ax_violin.plot([],[], linewidth=10, c="gray", label='Ground truth')
    ax_violin.legend(fontsize=15)
    ax_violin.set_ylabel('Drug Scores', fontsize=19)
    ax_violin.set_xlabel('')
    ax_violin.legend(loc=1, fontsize=19,ncol=2)
    ax_violin.tick_params(axis='x', rotation=40)
    ax_violin.set_xticks([])  # Remove the x-ticks
    ax_violin.set_xticklabels([])  # Remove the x-tick labels

    # MSE bar plot on the bottom
    methods = list(mses.keys())
    mse_values = list(mses.values())
    ax_bar.bar(methods, mse_values, color=[colors[mod] for mod in (models)])
    ax_bar.set_ylabel('MSE', fontsize=19)
    #ax_bar.set_xlabel('Method', fontsize=18)
    ax_bar.tick_params(axis='x', rotation=30, labelsize=19)

    # Align the x-ticks and make sure both plots share the same x-axis
    ax_bar.set_xticks(range(len(methods)))
    ax_bar.set_xticklabels(methods)

    # Save and show plot
    plt.tight_layout()
    ax_violin.text(-1.25,1.45, "(b)", fontsize=20)
    plt.savefig(os.path.join(OUT_DIR, "drug_scores_violin_mse.png"), bbox_inches='tight', dpi=300)

    plt.close()


    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    from matplotlib.gridspec import GridSpec
    from sklearn.metrics import explained_variance_score


    size_dots = 2

    custom_font_size = 20
    custom_fontsize_labelsaxis = 16
    fontsizelegend = 18
    tit = ['scClone2DR', 'FM', 'NN', 'Dual bulk', 'Bulk', 'Baseline',  'FM model (true props)', 'NN model (true props)']
    # Sample data (replace with your actual data)
    #colors = res['us']['colors']

    # Create subplots
    fig = plt.figure(figsize=(16, 13))
    gs = GridSpec(2, 3, width_ratios=[1, 1, 1])

    # Plot for model 1
    ax00 = fig.add_subplot(gs[0, 0])
    fc_true = res['us'].fold_change_true
    fc_pred = res['us'].fold_change_pred
    corr = np.round(np.corrcoef(fc_true, fc_pred)[0,1]**2, 3)
    rsquare = np.round(explained_variance_score(fc_true, fc_pred), 3)
    ax00.scatter(fc_true, fc_pred, label='{0}'.format(np.round(corr,3)), s=size_dots, c=colors['us'])
    ax00.set_title('{0}'.format(tit[0]), fontsize=custom_font_size)
    ax00.set_xlabel('Fold change (ground truth)', fontsize=custom_fontsize_labelsaxis)
    ax00.set_ylabel('Fold change (predicted)', fontsize=custom_fontsize_labelsaxis)
    m = min([np.min(fc_pred),np.min(fc_true)])
    ax00.set_ylim(min([np.min(fc_pred),np.min(fc_true)]), max([np.max(fc_pred),np.max(fc_true)]))
    ax00.text(m+0.02,m+0.02, "Explained variance: {0}".format(corr,rsquare), fontsize=14)
    #ax00.legend(title='Correlation', fontsize=fontsizelegend)
    # plt.hlines(0,xmin=lim_min,xmax=lim_max)
    # plt.vlines(0,ymin=lim_min,ymax=lim_max)
    # plt.xlim(lim_min,lim_max)
    # plt.ylim(lim_min,lim_max)

    # Plot for model 2
    ax01 = fig.add_subplot(gs[0, 1])
    fc_true = res['us'].fold_change_true
    fc_pred = res['fm'].fold_change_pred.numpy().flatten()
    corr = np.round(np.corrcoef(fc_true, fc_pred)[0,1]**2, 3)
    rsquare = np.round(explained_variance_score(fc_true, fc_pred), 3)
    ax01.scatter(fc_true, fc_pred, label='{0}'.format(np.round(corr,3)), s=size_dots, c=colors['fm'])
    ax01.set_title('{0}'.format(tit[1]), fontsize=custom_font_size)
    ax01.set_xlabel('Fold change (ground truth)', fontsize=custom_fontsize_labelsaxis)
    ax01.set_ylabel('Fold change (predicted)', fontsize=custom_fontsize_labelsaxis)
    m = min([np.min(fc_pred),np.min(fc_true)])
    ax01.set_ylim(min([np.min(fc_pred),np.min(fc_true)]), max([np.max(fc_pred),np.max(fc_true)]))
    ax01.set_xlim(min([np.min(fc_pred),np.min(fc_true)]), max([np.max(fc_pred),np.max(fc_true)]))
    ax01.text(m+0.02,m+0.02, "Explained variance: {0}".format(corr,rsquare), fontsize=14)
    #ax01.legend(title='Correlation', fontsize=fontsizelegend)

    # Plot for model 3
    ax10 = fig.add_subplot(gs[0, 2])
    fc_true = res['us'].fold_change_true
    fc_pred = res['nn'].fold_change_pred.numpy().flatten()
    corr = np.round(np.corrcoef(fc_true, fc_pred)[0,1]**2, 3)
    rsquare = np.round(explained_variance_score(fc_true, fc_pred), 3)
    ax10.scatter(fc_true, fc_pred, label='{0}'.format(np.round(corr,3)), s=size_dots, c=colors['nn'])
    ax10.set_title('{0}'.format(tit[2]), fontsize=custom_font_size)
    ax10.set_xlabel('Fold change (ground truth)', fontsize=custom_fontsize_labelsaxis)
    ax10.set_ylabel('Fold change (predicted)', fontsize=custom_fontsize_labelsaxis)
    m = min([np.min(fc_pred),np.min(fc_true)])
    ax10.set_ylim(min([np.min(fc_pred),np.min(fc_true)]), max([np.max(fc_pred),np.max(fc_true)]))
    ax10.set_xlim(min([np.min(fc_pred),np.min(fc_true)]), max([np.max(fc_pred),np.max(fc_true)]))
    ax10.text(m+0.02,m+0.02, "Explained variance: {0}".format(corr,rsquare), fontsize=14)
    #ax10.legend(title='Correlation', fontsize=fontsizelegend)

    # Plot for model 4
    ax11 = fig.add_subplot(gs[1, 0])
    fc_true = res['us'].fold_change_true
    fc_pred = res['bimodal'].fold_change_pred
    corr = np.round(np.corrcoef(fc_true, fc_pred)[0,1]**2, 3)
    rsquare = np.round(explained_variance_score(fc_true, fc_pred), 3)
    ax11.scatter(fc_true, fc_pred, label='{0}'.format(np.round(corr,3)), s=size_dots, c=colors['bimodal'])
    ax11.set_title('{0}'.format(tit[3]), fontsize=custom_font_size)
    ax11.set_xlabel('Fold change (ground truth)', fontsize=custom_fontsize_labelsaxis)
    ax11.set_ylabel('Fold change (predicted)', fontsize=custom_fontsize_labelsaxis)
    m = min([np.min(fc_pred),np.min(fc_true)])
    ax11.set_ylim(min([np.min(fc_pred),np.min(fc_true)]), max([np.max(fc_pred),np.max(fc_true)]))
    ax11.set_xlim(min([np.min(fc_pred),np.min(fc_true)]), max([np.max(fc_pred),np.max(fc_true)]))
    ax11.text(m+0.02,m+0.02, "Explained variance: {0}".format(corr,rsquare), fontsize=14)
    #ax11.legend(title='Correlation', fontsize=fontsizelegend)


    # Plot for model 5
    ax12 = fig.add_subplot(gs[1, 1])
    fc_true = res['us'].fold_change_true
    fc_pred = res['bulk'].fold_change_pred
    corr = np.round(np.corrcoef(fc_true, fc_pred)[0,1]**2, 3)
    rsquare = np.round(explained_variance_score(fc_true, fc_pred), 3)
    ax12.scatter(fc_true, fc_pred, label='{0}'.format(np.round(corr,3)), s=size_dots, c=colors['bulk'])
    ax12.set_title('{0}'.format(tit[4]), fontsize=custom_font_size)
    ax12.set_xlabel('Fold change (ground truth)', fontsize=custom_fontsize_labelsaxis)
    ax12.set_ylabel('Fold change (predicted)', fontsize=custom_fontsize_labelsaxis)
    m = min([np.min(fc_pred),np.min(fc_true)])
    ax12.set_ylim(min([np.min(fc_pred),np.min(fc_true)]), max([np.max(fc_pred),np.max(fc_true)]))
    ax12.set_xlim(min([np.min(fc_pred),np.min(fc_true)]), max([np.max(fc_pred),np.max(fc_true)]))
    ax12.text(m+0.02,m+0.02, "Explained variance: {0}".format(corr,rsquare), fontsize=14)
    #ax12.legend(title='Correlation', fontsize=fontsizelegend)

    # Plot for model 6
    ax02 = fig.add_subplot(gs[1, 2])
    fc_true = res['us'].fold_change_true
    fc_pred = res['base'].fold_change_pred
    corr = np.round(np.corrcoef(fc_true, fc_pred)[0,1]**2, 3)
    rsquare = np.round(explained_variance_score(fc_true, fc_pred), 3)
    ax02.scatter(fc_true, fc_pred, label='{0}'.format(np.round(corr,3)), s=size_dots, c=colors['base'])
    ax02.set_title('{0}'.format(tit[5]), fontsize=custom_font_size)
    m = min([np.min(fc_pred),np.min(fc_true)])
    ax02.set_xlabel('Fold change (ground truth)', fontsize=custom_fontsize_labelsaxis)
    ax02.set_ylabel('Fold change (predicted)', fontsize=custom_fontsize_labelsaxis)
    ax02.set_ylim(min([np.min(fc_pred),np.min(fc_true)]), max([np.max(fc_pred),np.max(fc_true)]))
    ax02.set_xlim(min([np.min(fc_pred),np.min(fc_true)]), max([np.max(fc_pred),np.max(fc_true)]))
    ax02.text(m+0.02,m+0.02, "Explained variance: {0}".format(corr,rsquare), fontsize=14)
    #ax02.legend(title='Correlation', fontsize=fontsizelegend)

    # Create a colorbar
    # Adjust layout
    plt.tight_layout()

    ax00.text(-0.3,0.083, "(a)", fontsize=20)
    plt.savefig(os.path.join(OUT_DIR, "fold_change_scatter.png"), bbox_inches='tight', dpi=300)

    # close the plot
    plt.close()



if __name__ == "__main__":
    for response_misspec in ["None", "Hill", "Saturating", "Threshold"]: # "None", "Hill",
        for noise_misspec in ["None", "Heteroscedastic", "ZeroInflated"]:
            for swap_frac in [0., 0.2, 0.4]:
                if noise_misspec in ["None", "HeavyTail"]:
                    main(response_misspec, noise_misspec, swap_frac=swap_frac)


                else:
                    for dispersion in [100]:
                        main(response_misspec, noise_misspec, dispersion=dispersion, swap_frac=swap_frac)


