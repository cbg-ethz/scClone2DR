import numpy as np
import torch
import matplotlib.pyplot as plt
from ..utils import *

import seaborn as sns
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec


def scatter_score(data_train, data_test,means_train, means_test, show_test=True, show_train=True, savefig=True):
    name = ''
    fig, ax = plt.subplots()

    if not(show_train):
        name='test'
    else:
        ax.scatter(data_train['log_scores'].reshape(-1), means_train.reshape(-1), c='orange', marker='.', label='Training samples')

    if not(show_test):
        name='train'
    else:
        ax.scatter(data_test['log_scores'].reshape(-1), means_test.reshape(-1), c='blue', marker='x', alpha=0.2, label='Test samples')

    lims = [
        np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
        np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
    ]

    # now plot both limits against eachother
    ax.plot(lims, lims, 'k-', alpha=0.75, zorder=0)
    ax.set_aspect('equal')
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.vlines(0, lims[0], lims[1], color='black', linestyle=(0, (3, 5, 1, 5)))
    ax.hlines(0, lims[0], lims[1], color='black', linestyle=(0, (3, 5, 1, 5)))
    plt.xlabel('Observed log score', fontsize=13)
    plt.ylabel('Predicted log score', fontsize=13)
    plt.legend()
    if savefig:
        if data_train['simulated_data']:
            plt.savefig('/data/users/quentin/Paper/figures/simu_data_'+name+'scatter_scores.png', dpi=250)
        else:
            if use_scatrex:
                plt.savefig('/data/users/quentin/Paper/figures/real_data_'+name+'scatter_score_scatrex.png', dpi=250)
            else:
                plt.savefig('/data/users/quentin/Paper/figures/real_data_'+name+'scatter_score_phenograph.png', dpi=250)
    plt.show()
    
    
    
def check_dirmulti_parametrization(data_train):
    fano_h = [] 
    fano_t = []
    for i in range(30):
        k = torch.sum(data_train['masks']['C'][:,i])
        var = torch.std(data_train['n0_c'][:k,i])**2
        mean = torch.mean(data_train['n0_c'][:k,i])
        fano_h.append(var/mean)
        var = torch.std(data_train['n_c'][:k,i]- data_train['n0_c'][:k,i])**2
        mean = torch.mean(data_train['n_c'][:k,i]-data_train['n0_c'][:k,i])
        fano_t.append(var/mean)

    plt.figure(1)
    plt.scatter(np.log(fano_h), np.log(fano_t))
    ls = np.linspace(0,np.max(np.log(fano_t)), 100)
    plt.plot(ls,ls, c='black', linestyle='--', label='line $y=x$')
    plt.xlabel('$\log (\\frac{\mathrm{Var}(N^{C0}_i)}{\mathbb {E}[N^{C0}_{i}]})$', fontsize=16)
    plt.ylabel('$\log (\\frac{\mathrm{Var}(N^{C+}_{i})}{\mathbb{E}[N^{C+}_{i}]})$', fontsize=16)
    plt.legend(fontsize=13)
    plt.savefig('parametrization1.png', dpi=250, bbox_inches='tight')
    plt.show()
    # plt.figure()
    # plt.scatter(fano_h, fano_t)
    # ls = np.linspace(0,np.max(fano_t), 100)
    # plt.plot(ls,ls, c='black', linestyle='--')
    # #plt.title('$\\frac{m_0 }{v_0}$ for the different samples', fontsize=14)
    # plt.xlabel('fano_h', fontsize=14)
    # plt.ylabel('fano_t', fontsize=14)
    # plt.legend(fontsize=14)
    print('R squared in log: ', 1-np.mean((np.log(fano_t)-np.log(fano_h))**2)/(np.std(np.log(fano_t))**2))
    print('R squared: ', 1-np.mean((np.array(fano_t)-np.array(fano_h))**2)/(np.std(fano_t)**2))
    # plt.savefig('parametrization1.png', dpi=250, bbox_inches='tight')
    
    
def check_negbin_parametrization(data_train):
    quadra_h = [] 
    quadra_t = []
    for i in range(30):
        k = torch.sum(data_train['masks']['C'][:,i])
        var = torch.std(data_train['n0_c'][:k,i])**2
        mean = torch.mean(data_train['n0_c'][:k,i])
        quadra_h.append((var-mean)/(mean**2))
        var = torch.std(data_train['n_c'][:k,i]- data_train['n0_c'][:k,i])**2
        mean = torch.mean(data_train['n_c'][:k,i]-data_train['n0_c'][:k,i])
        quadra_t.append((var-mean)/(var+mean**2))

    plt.figure(1)
    plt.scatter(np.log(quadra_h), np.log(quadra_t))
    ls = np.linspace(np.min(np.log(quadra_t)),np.max(np.log(quadra_t)), 100)
    plt.plot(ls,ls, c='black', linestyle='--',label='line $y=x$')
    plt.xlabel('$\log (\\frac{\mathrm{Var}(N^{C0}_i)-\mathbb {E}[N^{C0}_{i}]}{\mathbb {E}[N^{C0}_{i}]^2})$', fontsize=16)
    plt.ylabel('$\log (\\frac{\mathrm{Var}(N^{C+}_{i})-  \mathbb {E}[N^{C+}_{i}]}{{\mathrm{Var}(N^{C+}_{i})+\mathbb{E}[N^{C+}_{i}]^2}})$', fontsize=16)
    #plt.title('$\\frac{m_0 }{v_0}$ for the different samples', fontsize=14)
    print('R squared: ', 1-np.mean((np.log(quadra_t)-np.log(quadra_h))**2)/(np.std(np.log(quadra_t))**2))
    print('R squared: ', 1-np.mean((np.array(quadra_t)-np.array(quadra_h))**2)/(np.std(quadra_t)**2))
    plt.legend(fontsize=13)
    plt.savefig('parametrization2.png', dpi=250, bbox_inches='tight')
    
    
def show_cells(data, data_sample):
    R,C,N,Kmax = data['R'], data['C'], data['N'], data['Kmax']
    for c in range(C):
        lsids = [i for i, el in enumerate(data['masks']['C'][c,:]) if el]
        if c==0:
            plt.scatter(lsids, data_sample['n0_c'][c,lsids], c='red', marker='+', label='Estimation')
            plt.scatter(lsids, data['n0_c'][c,lsids], c='orange', marker = "x", label='Ground truth')
        else:
            plt.scatter(lsids, data_sample['n0_c'][c,lsids], c='red', marker='+')
            plt.scatter(lsids, data['n0_c'][c,lsids], c='orange', marker = "x")
    plt.legend()
    #plt.title('Control wells', fontsize=13)
    plt.xlabel('Test samples', fontsize=13)
    plt.title('Number of healthy cells in control wells', fontsize=13)
    plt.show()
    
    
def show_fractions(data, data_sample, idxdrug=0):
    D,R,C,N = data['D'], data['R'], data['C'], data['N']
    Ypred_c = np.zeros((C,N))
    Ypred_r = np.zeros((R,N))
    if True:
        for c in range(C):
            lsids = [i for i, el in enumerate(data['masks']['C'][c,:]) if el]
            for i in lsids:
                Ypred_c[c,i] = 1- data_sample['n0_c'][c,i]/data['n_c'][c,i]
        for r in range(R):
            lsids = [i for i, el in enumerate(data['masks']['R'][r,idxdrug,:]) if el]
            for i in lsids:
                Ypred_r[r,i] = 1- data_sample['n0_r'][r,idxdrug,i]/data['n_r'][r,idxdrug,i]

    else:
        for i in range(N):
            for c in range(C):
                Ypred_c[c,i] = 1- pred_summary['n0_c']['mean'][c,i]/pred_summary['n_c']['mean'][c,i]
            for r in range(R):
                Ypred_r[r,i] = 1- pred_summary['n0_r']['mean'][r,idxdrug,i]/pred_summary['n_r']['mean'][r,idxdrug,i]
    Y_c = np.zeros((C,N))
    Y_r = np.zeros((R,N))
    for c in range(C):
        lsids = [i for i, el in enumerate(data['masks']['C'][c,:]) if el]
        for i in lsids:
            Y_c[c,i] = 1- data['n0_c'][c,i]/data['n_c'][c,i]
    for r in range(R):
        lsids = [i for i, el in enumerate(data['masks']['R'][r,idxdrug,:]) if el]
        for i in lsids:
            Y_r[r,i] = 1- data['n0_r'][r,idxdrug,i]/data['n_r'][r,idxdrug,i]
            
    for c in range(C):
        lsids = [i for i, el in enumerate(data['masks']['C'][c,:]) if el]
        if c==0:
            plt.scatter(lsids, Ypred_c[c,lsids], c='red', marker='+', label='Estimation')
            plt.scatter(lsids, Y_c[c,lsids], c='orange', marker = "x", label='Ground truth')
        else:
            plt.scatter(lsids, Ypred_c[c,lsids], c='red', marker='+')
            plt.scatter(lsids, Y_c[c,lsids], c='orange', marker = "x")
    plt.legend()
    plt.title('Control wells', fontsize=13)
    plt.xlabel('Training Samples', fontsize=13)
    plt.ylabel('Fraction of tumor cells', fontsize=13)
    plt.show()
    
    for r in range(R):
        lsids = [i for i, el in enumerate(data['masks']['R'][r,idxdrug,:]) if el]
        if r==0:
            plt.scatter(lsids, Ypred_r[r,lsids], c='red', marker='+', label='Estimation')
            plt.scatter(lsids, Y_r[r,lsids], c='orange', marker = "x", label='Ground truth')
        else:
            plt.scatter(lsids, Ypred_r[r,lsids], c='red', marker='+')
            plt.scatter(lsids, Y_r[r,lsids], c='orange', marker = "x")
    plt.legend()
    plt.title('Wells with drug', fontsize=13)
    plt.xlabel('Training samples', fontsize=13)
    plt.ylabel('Fraction of tumor cells', fontsize=13)
    plt.show()

    
    
    

    
def show_proportions(data, params_svi=None, Nplotmin=0, Nplot=10, savefig=None):
    # Comparing true proportions of cells, the RNA data and the proportions learned by the model
    R,C,N,Kmax = data['R'], data['C'], data['N'], data['Kmax']
    Nplot = min([N,Nplot])
    import plotly.graph_objects as go
    x = [[],[]]
    for i in range(Nplot):
        x[0].append("patient "+str(i))
        x[0].append("patient "+str(i))
        if data['simulated_data']:
            x[0].append("patient "+str(i))
            x[1].append("true prop")
        x[1].append("RNA prop")
        x[1].append("estimated prop")

    fig = go.Figure()
    for k in range(Kmax):

        yrna = [data['n_rna'][k,Nplotmin+i]/torch.sum(data['n_rna'][:,Nplotmin+i]) for i in range(Nplot)]
        yesti = params_svi['proportions'][:,k]
        #yesti = res['params_proportions'][:,k]/np.sum(res['params_proportions'], axis=1)#[pred_summary['n_rna']['mean'][k,i]/torch.sum(pred_summary['n_rna']['mean'][:,i]) for i in range(Nplot)]
        y = []
        for i in range(Nplot):
            if data['simulated_data']:
                ytrue = data['proportions'][Nplotmin:Nplotmin+Nplot,k]
                y.append(ytrue[i])
            y.append(yrna[i])
            y.append(yesti[Nplotmin+i])
        fig.add_bar(x=x,y=y)
    fig.update_layout(barmode="relative")
    if not(savefig is None):
        fig.write_image(savefig)
    # for i in range(N):
    #     fig.add_trace(go.Scatter(x=[0, (i+1)], y=[frac_healthy_train[i], frac_healthy_train[i]],mode='lines',marker_color='black'))
    fig.show()


    
def show_beta(data, params_svi, true_params=None, idxdrug=0):
    # Comparing the true regression vector and the one learned
    if data['simulated_data']:
        plt.plot(true_params['beta'][idxdrug,:], label="Ground truth")
        plt.plot(params_svi['beta'][idxdrug,:], label="Estimation")
    else:
        plt.scatter(range(len(params_svi['beta'][idxdrug,:])), np.sort(params_svi['beta'][idxdrug,:].reshape(-1)), label="Estimation")
    plt.title('Beta parameter', fontsize=13)
    plt.xlabel('Features', fontsize=13)
    plt.legend()
    plt.show()
    
    
    
    
    
    
    
    
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec


def survival_probabilities_relative_by_patient(data, ratio_pi, pi, cluster2clonelabel, cluster2cat, selected_drugs, df_info_cohort=None, sampleID=0, savefig=None):
    # Get unique cluster labels and assign a color to each
    unique_labels = np.unique(cluster2clonelabel)
    label_colors = plt.colormaps['tab20'](np.linspace(0, 1, len(unique_labels)))

    for d in range(data['D']):
        pi[d,:,:][~(data['masks']['RNA'])] = float('nan')
        pi[d,:,:][~(data['masks']['RNA'])] = float('nan')
        ratio_pi[d,:,:][~(data['masks']['RNA'])] = float('nan')
        ratio_pi[d,:,:][~(data['masks']['RNA'])] = float('nan')
        
    available_clusters = np.array([i for i in range(pi.shape[1]) if not(np.isnan(pi[0,i,sampleID]))])
    pi = pi[:,available_clusters,:]
    ratio_pi = ratio_pi[:,available_clusters,:]
    
    cluster2clonelabel = np.array(cluster2clonelabel)[available_clusters]
    cluster2cat = np.array(cluster2cat)[available_clusters]
    # Sort clusters based on their labels
    sorted_indices = np.argsort(cluster2clonelabel)
    sorted_pi = pi[:, sorted_indices, sampleID]
    sorted_ratio_pi = ratio_pi[:, sorted_indices, sampleID]


    signs = -np.ones(len(sorted_indices))
    for i, idx in enumerate(sorted_indices):
        if cluster2cat[idx]=='healthy':
            signs[i] = 1
    idxs_sort_drugs = np.argsort( np.sum(np.nan_to_num(sorted_ratio_pi) * signs[None,:], axis=1) )
    
    sorted_drugs = selected_drugs[idxs_sort_drugs]
    sorted_pi = sorted_pi[idxs_sort_drugs, :]
    sorted_ratio_pi = sorted_ratio_pi[idxs_sort_drugs, :]

    

    # Create a color array for the y-axis labels, formatted for display
    row_colors = np.array([label_colors[np.where(unique_labels == label)[0][0]] for label in np.array(cluster2clonelabel)[sorted_indices]])

    # Reshape to a 2D array of RGB colors (n_clusters, 3) -> (n_clusters, 1, 3) for displaying as an image
    row_colors = row_colors.reshape(-1, 1, 4)  # Change 3 to 4 if the color has an alpha channel (RGBA)

    # Create the main plot with two subplots: one for the color bar and one for the heatmap
    fig = plt.figure(layout="constrained", figsize=(15,8))
    gs = GridSpec(1, 2, figure=fig, width_ratios=[0.05, 1])

    # Plot the color bar on the left using the label colors
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(row_colors, aspect='auto', interpolation='nearest')
    ax1.axis('off')  # Remove the axis for the color bar

    # Plot the heatmap using seaborn
    ax2 = fig.add_subplot(gs[0, 1])
    from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

    # Define the custom colormap from red to gray to green
    colors = [(1, 0, 0),  # Red
              (0.5, 0.5, 0.5),  # Gray
              (0, 0, 1)]  # Blue

    custom_cmap = LinearSegmentedColormap.from_list("RedGrayBlue", colors)
    vmin = min([0.9,np.min(np.nan_to_num(sorted_ratio_pi))])
    vmax = max([1.1,np.max(np.nan_to_num(sorted_ratio_pi))])
    norm = TwoSlopeNorm(vmin=vmin, vmax=vmax, vcenter=1)
    sns.heatmap(sorted_ratio_pi.T, cmap=custom_cmap, norm=norm, cbar=True, ax=ax2, yticklabels=False)

    # Retrieve the colorbar from the heatmap
    colorbar = ax2.collections[0].colorbar
    # Set custom ticks for the colorbar (e.g., 10 ticks)
    ticks = np.linspace(vmin, 1, num=5)
    ticksbis = np.linspace(1, vmax, num=5)

    ticks = np.concatenate((ticks, ticksbis))
    colorbar.set_ticks(ticks)
    # Optionally, set custom labels for the ticks (formatted to 2 decimal places)
    colorbar.set_ticklabels([f'{tick:.2f}' for tick in ticks])
    
    # Set the y-axis labels to the sorted cluster indices
    ax2.set_yticks(np.arange(len(sorted_indices)) + 0.5)

    # Custom legend for cluster labels
    handles = [mpatches.Patch(color=label_colors[i], label=f'Cluster {label}') for i, label in enumerate(unique_labels)]
    ax2.legend(handles=handles, bbox_to_anchor=(0., 1.2), loc='upper right', borderaxespad=0., fontsize=13)
    ax2.set_xticks(np.arange(len(sorted_drugs)))
    ax2.set_xticklabels(sorted_drugs, rotation=90, fontsize=11)

    # Optional: If you have patient labels
    # ax_heatmap.set_xticks(ticks=np.arange(sorted_pi.shape[1]))
    # ax_heatmap.set_xticklabels(patient_labels, rotation=90)

    # Show the plot
    plt.suptitle(f"Heatmap for sample: {sampleID}", x=0.5, y=0.94, ha='center', fontsize=18)

    if not(savefig is None):
        plt.savefig(savefig, dpi=250, bbox_inches='tight')
    plt.show()
    
    
def survival_probabilities_relative_by_patient_optimized(data, pi, props, cluster2clonelabel, cluster2cat, cat2clusters, selected_drugs, df_info_cohort=None, sampleID=0, savefig=None):
    from copy import deepcopy
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
    from matplotlib.gridspec import GridSpec
    
    ratio_pi = deepcopy(pi)
    for idxdrug in range(pi.shape[0]):
        for sampleid in range(pi.shape[2]):
            learned_props = props[sampleid, :]
            healthy_survival = np.sum(
                np.nan_to_num(pi[idxdrug, :, sampleid] * learned_props)[cat2clusters['healthy']]
            )
            healthy_survival /= np.sum(learned_props[cat2clusters['healthy']])
            ratio_pi[idxdrug, :, sampleid] /= healthy_survival

    scores = np.sum(
        np.nan_to_num(ratio_pi)[:, cat2clusters['tumor'], :] * 
        (props[:, cat2clusters['tumor']].T)[None, :, :], axis=1
    )
    scores /= np.tile((np.sum(props[:, cat2clusters['tumor']], axis=1)).reshape(1, -1), (ratio_pi.shape[0], 1))
    scores = scores[:, sampleID]
    scores = np.clip(scores, a_min=0, a_max=10)

    for d in range(data['D']):
        pi[d, :, :][~(data['masks']['RNA'])] = float('nan')
        ratio_pi[d, :, :][~(data['masks']['RNA'])] = float('nan')

    available_clusters = np.array([
        i for i in range(pi.shape[1]) 
        if (not np.isnan(pi[0, i, sampleID]) and (cluster2cat[i] != 'healthy'))
    ]).astype(int)
    pi = pi[:, available_clusters, :]
    ratio_pi = ratio_pi[:, available_clusters, :]

    learned_props = props[sampleID, available_clusters]
    cluster2clonelabel = np.array(cluster2clonelabel)[available_clusters]
    cluster2cat = np.array(cluster2cat)[available_clusters]
    
    sorted_indices = np.argsort(learned_props)
    cluster2cat = cluster2cat[sorted_indices]
    learned_props = learned_props[sorted_indices]
    cluster2clonelabel = np.array(cluster2clonelabel)[sorted_indices]
    sorted_pi = pi[:, sorted_indices, sampleID]
    sorted_ratio_pi = ratio_pi[:, sorted_indices, sampleID]

    idxs_sort_drugs = np.argsort(scores)
    sorted_drugs = selected_drugs[idxs_sort_drugs]
    sorted_pi = sorted_pi[idxs_sort_drugs, :]
    sorted_ratio_pi = sorted_ratio_pi[idxs_sort_drugs, :]
    sorted_scores = scores[idxs_sort_drugs]

    label_colors = plt.colormaps['Oranges'](np.linspace(0, 1, 100))
    row_colors = np.array([label_colors[int(prop * 100)] for prop in learned_props])
    row_colors = row_colors.reshape(-1, 1, 4)

    fig = plt.figure(layout="constrained", figsize=(15, 8))
    gs = GridSpec(2, 2, figure=fig, height_ratios=[1, 0.1], width_ratios=[0.05, 1])

    ax1 = fig.add_subplot(gs[0, 0])
    # Replace ax1 with a horizontal bar chart
    ax1.clear()
    ax1.barh(
        y=np.arange(len(learned_props)), 
        width=learned_props, 
        color='orange', 
        edgecolor='black', 
        align='center', 
        height=0.9
    )
    ax1.set_yticks(np.arange(len(learned_props)))
    ax1.set_yticklabels([])
    ax1.invert_yaxis()  # Invert y-axis to match heatmap row order
    ax1.set_xlim(0, max(learned_props) * 1.1)  # Add padding to bar chart
    ax1.xaxis.set_label_position('top')
    ax1.xaxis.tick_top()
    ax1.set_xlabel("Cluster size", fontsize=14, loc='left')
    #ax1.tick_params(axis='x', labelsize=13)
    ax1.set_xticks([])
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    for i, prop in enumerate(learned_props):
        ax1.text(
            prop * 1.02,  # Position the text slightly to the right of the bar
            i,  # Align with the bar
            f"{prop:.2f}",  # Display the value with 2 decimal places
            va='center',  # Vertically center the text
            fontsize=12,  # Font size for readability
            color='black'  # Text color
        )


    ax2 = fig.add_subplot(gs[0, 1])
    colors = [(0, 0, 1), (0.5, 0.5, 0.5), (1, 0, 0)]
    custom_cmap = LinearSegmentedColormap.from_list("terrain", colors)
    vmin = min([0.9, np.min(np.nan_to_num(sorted_ratio_pi))])
    vmax = max([1.1, np.max(np.nan_to_num(sorted_ratio_pi))])
    norm = TwoSlopeNorm(vmin=vmin, vmax=vmax, vcenter=1)
    sns.heatmap(sorted_ratio_pi.T, cmap=custom_cmap, norm=norm, cbar=True, alpha=0.7, ax=ax2, yticklabels=False)
    ax2.tick_params(axis='x', which='both', bottom=False, labelbottom=False)
    ax2.set_ylabel('Tumor clusters', fontsize=14)
    colorbar = ax2.collections[0].colorbar
    colorbar.set_label("Relative cluster\nsurvival probability", fontsize=12, labelpad=10)

    
    ax3 = fig.add_subplot(gs[1, 1], sharex=ax2)
    idxs = np.where(sorted_scores<=1)[0]
    ax3.plot(np.arange(len(sorted_scores))[idxs], sorted_scores[idxs], marker='o', color='blue')
    idxs = np.where(sorted_scores>1)[0]
    ax3.plot(np.arange(len(sorted_scores))[idxs], sorted_scores[idxs], marker='o', color='red')
    ax3.set_xticks([])
    ax3.set_xticks(np.arange(len(sorted_drugs)))
    ax3.set_xticklabels(sorted_drugs, rotation=90, fontsize=13)
    ax3.set_ylabel("Relative tumor\nsurvival probability", fontsize=14)
    ax3.grid(axis='y', linestyle='--', alpha=0.7)


    plt.suptitle(f"Heatmap for sample: {sampleID}", ha='center', fontsize=18)

    if savefig:
        plt.savefig(savefig, dpi=250, bbox_inches='tight')
    plt.show()
    
def survival_probabilities_relative(data, ratio_pi, pi, cluster2clonelabel, df_info_cohort=None, idxdrug=0, drug_name=None, savefig=None, filtersampleidxs=None):

    for d in range(data['D']):
        pi[d,:,:][~(data['masks']['RNA'])] = float('nan')
        pi[d,:,:][~(data['masks']['RNA'])] = float('nan')
        ratio_pi[d,:,:][~(data['masks']['RNA'])] = float('nan')
        ratio_pi[d,:,:][~(data['masks']['RNA'])] = float('nan')
        
    if filtersampleidxs is None:
        filtersampleidxs = np.arange(0, pi.shape[2], 1)
    # Sort clusters based on their labels
    sorted_indices = np.arange(len(cluster2clonelabel))#np.argsort(cluster2clonelabel)
    sorted_pi = pi[idxdrug, sorted_indices, :][:,filtersampleidxs]
    sorted_ratio_pi = ratio_pi[idxdrug, sorted_indices, :][:,filtersampleidxs]

    if not(df_info_cohort is None):
        # Sort samples by patient_id and tissue_type
        sample_names = df_info_cohort.index.values[filtersampleidxs]
        sorted_meta_data = df_info_cohort.sort_values(by=['patient_id', 'tissue_type'])
        sorted_sample_ids = sorted_meta_data.index
        sample2idx = {sample: idx for idx, sample in enumerate(sorted_sample_ids.values)}
        sorted_indices_patient = np.array([sample2idx[sample] for sample in sample_names])
        sorted_pi = sorted_pi[:, sorted_indices_patient]
        sorted_ratio_pi = sorted_ratio_pi[:, sorted_indices_patient]

    # Get unique cluster labels and assign a color to each
    unique_labels = np.unique(cluster2clonelabel)
    label_colors = plt.colormaps['tab20'](np.linspace(0, 1, len(unique_labels)))

    # Create a color array for the y-axis labels, formatted for display
    row_colors = np.array([label_colors[np.where(unique_labels == label)[0][0]] for label in np.array(cluster2clonelabel)[sorted_indices]])

    # Reshape to a 2D array of RGB colors (n_clusters, 3) -> (n_clusters, 1, 3) for displaying as an image
    row_colors = row_colors.reshape(-1, 1, 4)  # Change 3 to 4 if the color has an alpha channel (RGBA)

    if not(df_info_cohort is None):
        # Create the main plot with two subplots: one for the color bar and one for the heatmap
        fig = plt.figure(layout="constrained")
        gs = GridSpec(2, 2, figure=fig, width_ratios=[0.05, 1], height_ratios=[1, 0.05])
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1])
    else:
        # Create the main plot with two subplots: one for the color bar and one for the heatmap
        fig, (ax1, ax2) = plt.subplots(ncols=2, gridspec_kw={"width_ratios": [0.05, 1]}, figsize=(12, 8))

    # Plot the color bar on the left using the label colors
    ax1.imshow(row_colors, aspect='auto', interpolation='nearest')
    ax1.axis('off')  # Remove the axis for the color bar

    # Plot the heatmap using seaborn
    from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

    # Define the custom colormap from red to gray to green
    colors = [(1, 0, 0),  # Red
              (0.5, 0.5, 0.5),  # Gray
              (0, 0, 1)]  # Blue

    custom_cmap = LinearSegmentedColormap.from_list("RedGrayBlue", colors)
    idxsinf = np.isinf(sorted_ratio_pi)
    vmin = min([0.9,np.min(np.nan_to_num(sorted_ratio_pi[~idxsinf]))])
    vmax = max([1.1,np.max(np.nan_to_num(sorted_ratio_pi[~idxsinf]))])
    sorted_ratio_pi[idxsinf] = vmax

    norm = TwoSlopeNorm(vmin=np.min(np.nan_to_num(sorted_ratio_pi)), vmax=np.max(np.nan_to_num(sorted_ratio_pi)), vcenter=1)
    sns.heatmap(sorted_ratio_pi, cmap=custom_cmap, norm=norm, cbar=True, ax=ax2, yticklabels=False)

    # Set the y-axis labels to the sorted cluster indices
    ax2.set_yticks(np.arange(len(sorted_indices)) + 0.5)
#     if cluster2clonetype is None:
#         ax2.set_yticklabels(np.array(cluster2clonelabel)[sorted_indices], rotation=0)
#     else:
#         ax2.set_yticklabels(np.array(cluster2clonetype)[sorted_indices], rotation=0)

    # Custom legend for cluster labels
    handles = [mpatches.Patch(color=label_colors[i], label=f'Cluster {label}') for i, label in enumerate(unique_labels)]
    
    if df_info_cohort is None:
        ax2.legend(handles=handles, bbox_to_anchor=(0., 1.1), loc='upper right', borderaxespad=0.)
    else:
        ax2.legend(handles=handles, bbox_to_anchor=(0., 1.2), loc='upper right', borderaxespad=0.)
    ax2.set_xticks([])

    # Optional: If you have patient labels
    # ax_heatmap.set_xticks(ticks=np.arange(sorted_pi.shape[1]))
    # ax_heatmap.set_xticklabels(patient_labels, rotation=90)

    # Show the plot
    if not(drug_name is None):
        plt.suptitle(f"Heatmap for the drug: {drug_name}", x=0.5, y=0.92, ha='center')
    else:
        plt.suptitle(f"Heatmap for Drug Index {idxdrug}", x=0.5, y=0.92, ha='center')

    if not(df_info_cohort is None):
        ax3 = fig.add_subplot(gs[1, 1])
        # Get unique cluster labels and assign a color to each
        unique_labels = np.unique(sorted_meta_data['patient_id'].values)
        label_colors = plt.colormaps['tab20'](np.linspace(0, 1, len(unique_labels)))
        # Create a color array for the y-axis labels, formatted for display
        col_colors = np.array([label_colors[np.where(unique_labels == label)[0][0]] for label in np.array(sorted_meta_data['patient_id'].values)])

        # Reshape to a 2D array of RGB colors (n_clusters, 3) -> (n_clusters, 1, 3) for displaying as an image
        col_colors = col_colors.reshape(1, -1, 4)  # Change 3 to 4 if the color has an alpha channel (RGBA)
        ax3.imshow(col_colors, aspect='auto', interpolation='nearest')
        #ax3.axis('off')  # Remove the axis for the color bar
        ax3.set_xlabel('Samples')
        ax3.set_xticks([])
        ax3.set_yticks([])
    else:
        ax2.set_xlabel('Samples')

    if not(savefig is None):
        plt.savefig(savefig, dpi=250, bbox_inches='tight')
    plt.show()    

    
def survival_probabilities(data, pi, cluster2clonelabel, df_info_cohort=None, idxdrug=0, drug_name=None, clustername='subclone', savefig=None):
    # Comparing true survival probabilities and the one estimated
    
    label2name = {'healthy':'Non-malignant', 'tumor':'Tumor'}
    
    for d in range(data['D']):
        pi[d,:,:][~(data['masks']['RNA'])] = float('nan')
        pi[d,:,:][~(data['masks']['RNA'])] = float('nan')
    
    # Sort clusters based on their labels
    sorted_indices = np.arange(len(cluster2clonelabel))#np.argsort(cluster2clonelabel)
    sorted_pi = pi[idxdrug, sorted_indices, :]
    
    if not(df_info_cohort is None):
        # Sort samples by patient_id and tissue_type
        sample_names = df_info_cohort.index.values
        sorted_meta_data = df_info_cohort.sort_values(by=['patient_id', 'tissue_type'])
        sorted_sample_ids = sorted_meta_data.index
        sample2idx = {sample: idx for idx, sample in enumerate(sorted_sample_ids.values)}
        sorted_indices_patient = np.array([sample2idx[sample] for sample in sample_names])
        sorted_pi = sorted_pi[:, sorted_indices_patient]
    
    # Get unique cluster labels and assign a color to each
    unique_labels = np.unique(cluster2clonelabel)
    label_colors = plt.colormaps['tab20'](np.linspace(0, 1, len(unique_labels)))
    
    # Create a color array for the y-axis labels, formatted for display
    row_colors = np.array([label_colors[np.where(unique_labels == label)[0][0]] for label in np.array(cluster2clonelabel)[sorted_indices]])
    
    # Reshape to a 2D array of RGB colors (n_clusters, 3) -> (n_clusters, 1, 3) for displaying as an image
    row_colors = row_colors.reshape(-1, 1, 4)  # Change 3 to 4 if the color has an alpha channel (RGBA)
    
    if not(df_info_cohort is None):
        # Create the main plot with two subplots: one for the color bar and one for the heatmap
        fig = plt.figure(layout="constrained")
        gs = GridSpec(2, 2, figure=fig, width_ratios=[0.05, 1], height_ratios=[1, 0.05])
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1])

    else:
        # Create the main plot with two subplots: one for the color bar and one for the heatmap
        fig, (ax1, ax2) = plt.subplots(ncols=2, gridspec_kw={"width_ratios": [0.05, 1]}, figsize=(12, 8))

    # Plot the color bar on the left using the label colors
    ax1.imshow(row_colors, aspect='auto', interpolation='nearest')
    ax1.axis('off')  # Remove the axis for the color bar
    
    # Plot the heatmap using seaborn
    sns.heatmap(sorted_pi, cmap='viridis', cbar=True, ax=ax2, yticklabels=False)
    
    # Set the y-axis labels to the sorted cluster indices
    ax2.set_yticks(np.arange(len(sorted_indices)) + 0.5)

    # Custom legend for cluster labels
    handles = [mpatches.Patch(color=label_colors[i], label=f'{label2name[label]} {clustername}') for i, label in enumerate(unique_labels)]
    ax2.legend(handles=handles, bbox_to_anchor=(1, 1.1), loc='upper right', borderaxespad=0.)
    ax2.set_xticks([])

    # Optional: If you have patient labels
    # ax_heatmap.set_xticks(ticks=np.arange(sorted_pi.shape[1]))
    # ax_heatmap.set_xticklabels(patient_labels, rotation=90)
    
    # Show the plot
    if not(drug_name is None):
        plt.suptitle(f"Heatmap for the drug: {drug_name}")
    else:
        plt.suptitle(f"Heatmap for Drug Index {idxdrug}")

    if not(df_info_cohort is None):
        ax3 = fig.add_subplot(gs[1, 1])
        # Get unique cluster labels and assign a color to each
        unique_labels = np.unique(sorted_meta_data['patient_id'].values)
        label_colors = plt.colormaps['tab20'](np.linspace(0, 1, len(unique_labels)))
        # Create a color array for the y-axis labels, formatted for display
        col_colors = np.array([label_colors[np.where(unique_labels == label)[0][0]] for label in np.array(sorted_meta_data['patient_id'].values)])

        # Reshape to a 2D array of RGB colors (n_clusters, 3) -> (n_clusters, 1, 3) for displaying as an image
        col_colors = col_colors.reshape(1, -1, 4)  # Change 3 to 4 if the color has an alpha channel (RGBA)
        ax3.imshow(col_colors, aspect='auto', interpolation='nearest')
        #ax3.axis('off')  # Remove the axis for the color bar
        ax3.set_xlabel('Samples')
        ax3.set_xticks([])
        ax3.set_yticks([])
    if not(savefig is None):
        plt.savefig(savefig, dpi=250, bbox_inches='tight')
    plt.show()    

    
    
    
    
    
####### BEGIN: fraction of melanoma cells
def scatter_counts(data_train, data_sample_train, mode_x='drug', mode_y='drug', savefig=True, R2=None, corrcoeff=None, marker='x', figname='', label='', color_mode=None, ylabel='Predicted proportion'):
    name = ''
    hsv = plt.get_cmap('jet')
    fig, ax = plt.subplots()

    if color_mode is None:
        colors = 'blue'
    elif color_mode=='drug':
        colors = []
        ref_colors = hsv(np.linspace(0, 1.0, data_train['D']))
        colors = np.repeat(ref_colors, data_train['N'], axis=0)
    elif color_mode=='patient':
        ref_colors = hsv(np.linspace(0, 1.0, data_train['N']))
        colors = np.tile(ref_colors, (data_train['D'],1))
    
    if mode_y=='drug':
        frac_r = 1. - data_sample_train['n0_r'] / data_train['n_r']    
        frac_mean_r_samp = torch.sum(data_train['masks']['R']*torch.nan_to_num(frac_r), dim=0) / torch.sum(data_train['masks']['R'], dim=0) 
        data_y = frac_mean_r_samp
        labely = 'Predicted fraction of tumor cells with drug'
        
    if mode_x=='drug':
        frac_r = 1. - data_train['n0_r'] / data_train['n_r']
        frac_mean_r = torch.sum(data_train['masks']['R']*torch.nan_to_num(frac_r), dim=0) / torch.sum(data_train['masks']['R'], dim=0) 
        data_x = frac_mean_r
        labelx = 'Observed fraction of tumor cells with drug'


    if mode_x=='control':
        frac_c = 1. - data_train['n0_c'] / data_train['n_c']
        frac_mean_c = (torch.sum(data_train['masks']['C']*torch.nan_to_num(frac_c), dim=0) / torch.sum(data_train['masks']['C'], dim=0)).numpy().repeat(data_train['D'])
        data_x = frac_mean_c
        labelx = 'Observed fraction of tumor cells in control'

        
    if mode_y=='control':
        frac_c = 1. - data_sample_train['n0_c'] / data_train['n_c']
        frac_mean_c = (torch.sum(data_train['masks']['C']*torch.nan_to_num(frac_c), dim=0) / torch.sum(data_train['masks']['C'], dim=0)).numpy().repeat(data_train['D'])
        data_y = frac_mean_c
        labely = 'Predicted fraction of tumor cells in control'
        
    if mode_x=='control_allreps':
        frac_c = 1. - data_train['n0_c'] / data_train['n_c']
        frac_c_tot = []
        for r in range(data_train['masks']['C'].shape[0]):
            for i in range(data_train['masks']['C'].shape[1]):
                if data_train['masks']['C'][r,i]:
                    frac_c_tot.append(frac_c[r,i])
        data_x = np.array(frac_c_tot)
        labelx = 'Observed fraction of tumor cells in control all replis'

        
    if mode_y=='control_allreps':
        frac_c = 1. - data_sample_train['n0_c'] / data_train['n_c']
        frac_c_tot = []
        for r in range(data_train['masks']['C'].shape[0]):
            for i in range(data_train['masks']['C'].shape[1]):
                if data_train['masks']['C'][r,i]:
                    frac_c_tot.append(frac_c[r,i])
        data_y = np.array(frac_c_tot)
        labely = 'Predicted fraction of tumor cells in control all replis'



    n = min(data_x.shape[-1],data_y.shape[-1])        
    if len(data_x.shape)==2:
        data_x = data_x[:,:n]
    else:
        data_x = data_x[:n]
    if len(data_y.shape)==2:
        data_y = data_y[:,:n]
    else:
        data_y = data_y[:n]
            
    data_x, data_y = data_x.reshape(-1), data_y.reshape(-1)

    if not(color_mode is None):
        idxs_nan = torch.isnan(data_y)
        colors = colors[~idxs_nan.numpy(),:]
        data_x = data_x[~idxs_nan]
        data_y = data_y[~idxs_nan]


    
    ax.scatter(data_x, data_y, c=colors, marker=marker, alpha=0.4, label = np.round(np.corrcoef(data_x, data_y)[0,1]**2, 4))
    ax.set_xlabel(labelx, fontsize=13)
    ax.set_ylabel(labely, fontsize=13)
        
    lims = [
        np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
        np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
    ]
    
    if not(R2 is None):
        plt.plot([], [], ' ', label="$R^2=$"+str(round(float(R2),3)))
        
    if not(corrcoeff is None):
        plt.plot([], [], ' ', label="$corrcoeff^2=$"+str(round(float(corrcoeff),5)))

    


    # now plot both limits against eachother
    ax.plot(lims, lims, 'k-', alpha=0.75, zorder=0)
    ax.set_aspect('equal')
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.vlines(0, lims[0], lims[1], color='black', linestyle=(0, (3, 5, 1, 5)))
    ax.hlines(0, lims[0], lims[1], color='black', linestyle=(0, (3, 5, 1, 5)))
    plt.legend(title='Correlation: ')  
    if figname!='':
        ax.set_title(figname)
        plt.savefig(figname+'.png', dpi=250, bbox_inches='tight')
    
def get_colors(color_mode, data_train):
#     all_color_modes = ['', 'drug', 'patient']
#     color_mode = all_color_modes[1]
    N = data_train['N']
    hsv = plt.get_cmap('jet')
    if color_mode=='drug':
        ref_colors = hsv(np.linspace(0, 1.0, data_train['D']))
        colors = []
        for d in range(data_train['D']):
            colors += [ref_colors[d] for i in range(N)]
        
    elif color_mode=='patient':
        
        ref_colors = hsv(np.linspace(0, 1.0, N))
        colors = []
        for d in range(data_train['D']):
            colors += [ref_colors[i] for i in range(N)]
    else:
        colors = np.array([['blue' for d in range(data_train['D'])] for i in range(data_train['N'])]).reshape(data_train['D']*data_train['N'])
    return colors

####### END: fraction of melanoma cells

