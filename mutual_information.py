import numpy as np
import os
import networkx as nx
import pandas as pd
from scipy.stats import chi2
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 16})
from time import time
import operator
from itertools import chain
import joblib
import argparse

def get_joint_and_marginals(data, col1, col2):
    # get the contingency table for these two columns in the data
    # the number of bins in the 2D histogram is the number of unique elements in each column
    Nxy,_,_ = np.histogram2d(np.array(data.iloc[:,col1]),np.array(data.iloc[:,col2]), 
                         bins= (len(data.iloc[:,col1].unique()), len(data.iloc[:,col2].unique())))
    # total number of samples
    N = float(Nxy.sum())
    # joint distribution P(x,y)
    Pxy = Nxy/N
    # marginals
    Px = Pxy.sum(axis=1)
    Py = Pxy.sum(axis=0)
    return Pxy, Px, Py, N
 
    
def H_mle(P):
    # skip all zero probabilities 
    idx = P > 0
    # return MLE entropy 
    return -(P[idx]*np.log2(P[idx])).sum()


def MI_mle(data, col1, col2):
    # Maximum likelihood MI estimate
    Pxy, Px, Py, Ntot = get_joint_and_marginals(data, col1, col2)
    return H_mle(Px) + H_mle(Py) - H_mle(Pxy) 


def H_mm(P, N):
    # Miller madow corrected estimate of entropy
    m = (P > 0).sum()
    return H_mle(P) + (m-1)/(2.0*N)


def MI_mle_miller_madow(data, col1, col2):
    Pxy, Px, Py, Ntot = get_joint_and_marginals(data, col1, col2)
    return H_mm(Px, Ntot) + H_mm(Py, Ntot) - H_mm(Pxy, Ntot)


def symm_uncertainty(data, col1, col2):
    Pxy, Px, Py, Ntot = get_joint_and_marginals(data, col1, col2)
    H_x = H_mm(Px, Ntot)
    H_y = H_mm(Py, Ntot)
    mi = H_x + H_y - H_mm(Pxy, Ntot)
    # the 1e-10 is a regularization that doesnt allow NaN 
    return mi, (2*mi)/(H_x + H_y + 1e-10)


def pearson_MI(data, col1, col2):
    Pxy, Px, Py, Ntot = get_joint_and_marginals(data, col1, col2)
    H_x = H_mm(Px, Ntot)
    H_y = H_mm(Py, Ntot)
    mi = H_x + H_y - H_mm(Pxy, Ntot)
    # the 1e-10 is a regularization that doesnt allow NaN 
    return mi, (mi)/(np.sqrt(H_x * H_y) + 1e-10)


def IQR_MI(data, col1, col2):
    Pxy, Px, Py, Ntot = get_joint_and_marginals(data, col1, col2)
    H_x = H_mm(Px, Ntot)
    H_y = H_mm(Py, Ntot)
    Hxy = H_mm(Pxy, Ntot)
    mi = H_x + H_y - Hxy
    # the 1e-10 is a regularization that doesnt allow NaN 
    return mi, mi / (Hxy + 1e-10)


color_dict_adult = {'#wgq+ss' : '#c7fdb5',
                    '#wgq+en' : '#75fd63',
                   '#wgq+es' : '#53fca1',
                   '#barriers+water': '#a2cffe',
                    '#access+water': '#a2cffe',
                   '#barriers+medical' : 'r',
                    '#access+medical' : 'r',
                   '#barriers+shelter': '#a57e52',
                    '#access+shelter': '#a57e52',
                   '#barriers+latrine' : '#e6daa6',
                    '#access+latrine' : '#e6daa6',
                   '#barriers+food' : '#fdaa48',
                    '#access+food' : '#fdaa48',
                   '#barriers+cash' : '#cea2fd',
                    '#access+cash' : '#cea2fd',
                    '#access+cash+income' : '#cea2fd',
                   '#fdiff': '#06b48b',
                    '#loc': '#d8dcd6',
                    '#personal': '#d46a7e',
                    '#fdiff+injury': '#b04e0f',
                    '#injury+war': '#960056',
                    '#fdiff+mental': '#74a662',
                    '#disability' : '#510ac9',
                   }

color_dict_child = {'#child+cfm' : '#c7fdb5',
                   '#child+edu' : '#a2cffe',
                    '#disability' : '#510ac9',
                   'other' : 'w'}


def color_name_map(node_ids, colnames, child = False):
    ''' builds a dictionary of mapping node number with its color
        based on the category that the name of the node belongs to.
        Dictionary of categories names as keys and colors as values 
        is "color_dict_adult" or "color_dict_child"
    '''
    node_colors = {}
    for i in node_ids:
        if not child:
            query = colnames[i].split('_')[0]
            if query in color_dict_adult.keys():
                node_colors[i] = color_dict_adult[query]
            else:
                node_colors[i] = 'w'
        else:
            # for children
            query = colnames[i].split('_')[0]
            if query in color_dict_child.keys():
                node_colors[i] = color_dict_child[query]
            else:
                node_colors[i] = 'w'
    return node_colors



def pairwise_mi_with_na_removal(data, method = 'symm_uncertainty', na_remove = True, thresh = 200):
    '''
    Computes Mutual Information between (discrete) variables in a dataframe using one of five available measures
    1. Maximum Likelihood Estimate (MLE)
    2. Miller Madow bias corrected MLE estimate
    3. Symmetric Uncertainty (normalized)
    4. Information Quality Ratio (IQR, normalized)
    5. Pearson MI (normalized)
    
    Params
    -------
            - data: pandas DataFrame containing discrete random variables
            - method: (str) one of four options {'symm_uncertainty', 'IQR', 'miller_madow',
                'pearson_MI'}. For any other input the default option is the maximum likelihood
                 MLE estimate.
                 For equations see
                 1. Introduction: https://en.wikipedia.org/wiki/Mutual_information
                 2. David Mackay http://www.inference.org.uk/itprnn/book.pdf
                 2. Clustering based on Mutual information: 
                     -- Ver Steeg et al. http://proceedings.mlr.press/v32/steeg14.pdf
            - na_remove: (bool) whether to remove samples (rows) with missing values
                from pairwise MI calculation
            - thresh: (int) minimum number of non-missing samples (rows) needed in both variables
                being compared in MI calculation
                
    Returns
    --------
            - edges: dictionary of edge numbers as keys and edges as values. 
                     Each edge 'i' is a tuple (node1, node2, M.I value) 
    '''
    # this dictionary will hold edges as tuples ()
    edges = {}
    # 
    k = 0
    for i in range(data.shape[1]):
        # if node i has only one unique value, skip it
        u = data.iloc[:,i].notna()
        unqs = data.loc[u, data.columns[i]].unique()
        if len(unqs) == 1:
            #print('.....dropped column %d , variable %s ......'%(i, data.columns[i]))
            continue
        for j in range(i,data.shape[1]):
            # do not consider self-loops
            if i == j:
                continue
            # if node j has only one unique value, skip it
            u = data.iloc[:,j].notna()
            unqs = data.loc[u, data.columns[j]].unique()
            if len(unqs) == 1:
                #print('.....dropped column %d , variable %s ......'%(j, data.columns[j]))
                continue
                
            if na_remove:
                # get all nan rows for node i and j
                notnanrows_i = np.where(data.iloc[:,i].notna())[0]
                notnanrows_j = np.where(data.iloc[:,j].notna())[0]
                intersec = np.intersect1d(notnanrows_i, notnanrows_j)
            else:
                data = data.mask(data.isna(), other = -1)
                intersec = np.arange(data.shape[0])
                
            if len(intersec) > thresh:
                if method == 'symm_uncertainty':
                    mi, mi_for_graph = symm_uncertainty(data.iloc[intersec,:], i, j)
                elif method == 'pearson':
                    mi, mi_for_graph = pearson_MI(data.iloc[intersec,:], i, j)
                elif method == 'miller_madow':
                    mi_for_graph = MI_mle_miller_madow(data.iloc[intersec,:], i, j)
                elif method == 'IQR':
                    mi, mi_for_graph = IQR_MI(data.iloc[intersec,:], i, j)
                else:
                    mi_for_graph = MI_mle(data.iloc[intersec,:], i, j)
                    
                if np.isnan(mi) or mi == 0.:
                    continue
                dof = (len(data.iloc[:,i].unique())-1) * (len(data.iloc[:,j].unique())-1)
                # get G statistic
                G, pval = get_G_statistic_pvalue(mi, dof, len(intersec))
                edges[k] = (i, j, mi, data.columns[i], data.columns[j], pval, len(intersec))
                k += 1
    return edges


def get_G_statistic_pvalue(mi, dof, N):
    ''' G statistic is Chi square distributed 
        dof : degrees of freedom
        mi : mutual information
        N : number of samples 
    '''
    G = 2 * N * mi
    pval = 1 - chi2.cdf(G, dof)
    return G, pval

def build_graph_v2(edges, thresh = 0.01):
    '''
    Builds a networkx Graph from a list of edges after thresholding them
    Parameters
    ----------
            - D: dictionary containing graph edges as keys (integers) 
                 Value for any key has form :  (node1,node2, edge_weight, node1_name, node2_name)
            - thresh:  A threshold to decide whether to keep this edge or not
    Returns
    ----------
            - G: networkx Graph object
            - weights: a list of weights that pass the threshold
            - edge_list: a list of tuples (node1 ,node2, {edge_attributes})
                         edge_attributes : weight
    '''
    G = nx.Graph()
    weights = []
    # loop over all edges
    for k,(node1,node2,w,_,_,_,_) in edges.items():
        if np.abs(w) >= thresh:
            # add node i and j if not already there
            if node1 not in G.nodes:
                G.add_node(node1)
            if node2 not in G.nodes:
                G.add_node(node2)
            # now add edge if not already present
            if (node1, node2) not in G.edges:
                G.add_edges_from([ (node1, node2, {'weight': np.abs(w)}) ])
            weights.append(np.abs(w))
            
    return G, weights



def draw_net(Gr, pos, weights, thresh, node_colors, colnames = None, link_weight = 0.5, circles_to_plot = None, 
             fig_size = (40,20), savepath = '', img_dpi = 100):
    '''
    Plots the networkX graph instance.
    
    Params
    -------
            - Gr : networkX graph instance
            - pos : (dict) node positions as determined by a networkX layout types (e.g. nx.spring_layout)
            - weights : (list) list of edge weights (floats)
            - thresh : (float) threshold used to prune edges in graph
            - node_colors : (dict) node index as keys and node colors as values
            - colnames : (list) strings of variable names
            - link_weight : (float) scaling term for edge thickness (higher means greater diversity in thickness)
            - circles_to_plot : (dict) for each node category as key, it has the color as value (for plotting legend)
            - fig_size : (tuple) figure size
            - savepath : (str) if non-empty (empty is default) uses this to save the plot as an image. Full path has 
            to exist and the format is identified from the name (e.g. if savepath = 'something.jpg', the format is jpeg)
            - img_dpi : (int) dpi image resolution, increasing it increases resolution and memory used.
    '''
    alignment = {'horizontalalignment': 'center', 'verticalalignment': 'baseline'}
    
    if circles_to_plot:
        fig, ax = plt.subplots(nrows = 1, ncols = 2, figsize = fig_size, gridspec_kw = {'width_ratios':[3, 1]})
        ax[1].set_xlim([0,  20])
        ax[1].set_ylim([0, 100])
        ax[1].spines['top'].set_visible(False)
        ax[1].spines['right'].set_visible(False)
        ax[1].spines['bottom'].set_visible(False)
        ax[1].spines['left'].set_visible(False)
        ax[1].get_xaxis().set_visible(False)
        ax[1].get_yaxis().set_visible(False)
        if circles_to_plot:
            i = 0
            for key, value in circles_to_plot.items():
                circle = plt.Circle(xy = (1.5, 85 - 4*i), radius = 1., color = value, fill = True)
                ax[1].add_patch(circle)
                ax[1].text(3.5, 84 - 4*i, key)
                i += 1
        a = ax[0]
    else:
        fig, a = plt.subplots(nrows = 1, ncols = 1, figsize = fig_size)
        
    fig.tight_layout()
    
    a.get_xaxis().set_visible(False)
    a.get_yaxis().set_visible(False)
    a.spines['top'].set_visible(False)
    a.spines['right'].set_visible(False)
    a.spines['bottom'].set_visible(False)
    a.spines['left'].set_visible(False)
    a.set_title('Graph with threshold %.5f'%(thresh))
    unique_weights = list(set(weights))
    nnodes = Gr.number_of_nodes()
    
    for i,w in enumerate(unique_weights):
        weighted_edges = [(node1,node2) for (node1,node2,edge_attr) in Gr.edges(data=True) if edge_attr['weight']==w]
        width = w * nnodes * link_weight/sum(weights)
        nx.draw_networkx_edges(Gr, pos, edgelist=weighted_edges,width=width, ax = a)
        if weighted_edges:
            nx.draw_networkx_nodes(Gr, pos, node_list = weighted_edges[0], with_labels=True, node_shape = 'o',
                         node_color = list(node_colors.values()), alpha = 0.2, node_size = 2000, ax = a)
    nx.draw_networkx_labels(Gr, pos, font_color='k', font_size=16, font_weight = 'normal', ax = a) 
        
    if len(savepath) > 0:
        plt.savefig(savepath, dpi = 100, format = savepath.split('.')[-1], bbox_inches='tight')
        plt.close()
        

        
def subselect_graph(edges, selected_nodes, only_these = False):
    # now take out all edges that contain at least one of the selected_nodes
    to_drop = []
    # cycle through edges
    for k,(node1, node2, w,_,_,_,_) in edges.items():
        if only_these:
            # both nodes must be a subset of selected_group nodes
            if (node1 not in selected_nodes) or (node2 not in selected_nodes):
                to_drop.append(k)
        else:
            # One of two nodes must be a subset of selected_group nodes
            if (node1 not in selected_nodes) and (node2 not in selected_nodes):
                to_drop.append(k)
    for k in to_drop:
        del edges[k]
    return edges
        
    

def get_top_mi_values(edges):
    '''
    Takes an edges dictionary of tuples (node1, node2, weight, node1name, node2name)
    and returns a sorted Data Frame of edges
    '''
    lst = []
    colnames = ['node1', 'node1name', 'node2', 'node2name',
                'mutual_info', 'p-value', 'sample_size']
    for k,(node1,node2,w,node1name,node2name,pval,N) in edges.items():
        lst.append([node1, node1name, node2, node2name, w, pval, N])
    weights = np.array([w[4] for w in lst])
    # sort descending order by mutual info (weight) value
    weight_idx = np.argsort(weights)[::-1]
    lst = [lst[i] for i in weight_idx]
    df = pd.DataFrame(lst, columns=colnames)
    return df



#########################################################################################

class MIGraphBuilder():
    '''
    Class that builds and plots a mutual information graph
    '''
    def __init__(self, MI_type = 'symm_uncertainty', na_remove = True, na_thresh = 200,
                MI_thresh = 0.1):
        self.MI_type = MI_type
        self.na_remove = na_remove
        self.na_thresh = na_thresh
        self.MI_thresh = MI_thresh
        
    def compute_graph(self, data, return_all = False, method = None, thresh = None):
        ''' computes pairwise Mutual Information measure '''
        start = time()
        # search for child token
        self.child = np.sum(data.columns.str.startswith('#child')) > 0
        self.colnames = data.columns
        if not thresh:
            thresh = self.na_thresh
        if not method:
            method = self.MI_type
        E = pairwise_mi_with_na_removal(data, method = method,
                                            na_remove = self.na_remove,
                                           thresh = thresh)
        self.edges = E
        end = time()
        print(' ..... MI graph computed in %.2f secs ...... '%(end-start))
        if return_all:
            return E, data
        
    def select_nodes(self, select_group = [], only_these = False):
        '''
        Utility function that allows user to select subsets of variables in the MI graph.
        
        Params
        -------
                - select_group : (list of strings) each string should contain the column headers
                to be selected
                - only_these : (bool, default = False) makes sure that only these nodes appear
                in graph. If False, only one of two nodes in an edge is required to be from amongst
                select_group.
        Returns
        -------
                - subset of graph (dict with tuples as edges)
        
        '''
        E = self.edges.copy()
        selected_nodes = []
        for s in select_group:
            selected_nodes.append(np.where(self.colnames.str.startswith(s))[0])
        selected_nodes = list(chain.from_iterable(selected_nodes))
        
        return subselect_graph(E, selected_nodes, only_these)
    
    def compute_centrality(self, Gr, centrality_type = 'degree'):
        '''
        This function computes the Graph "Centrality" of each node
        (here, a node is a variable or question in a survey dataset) using several different
        measures listed below. 
        
        Centrality measures essentially compute the importance of a node in terms of its 
        connectivity. For example, degree centrality of a node u is simply the fraction of nodes
        in the graph it is connected to.
        
        For more information:
        https://networkx.github.io/documentation/stable/reference/algorithms/centrality.html
        
        Params
        -------
                - Gr : a non-empty networkX graph instance
                - centrality_type : (str) Must be one of {'eigenvector', 'betweenness', 'katz',
                'current_flow', 'degree'}. Partial match (e.g. 'eigen') will work.
        Returns
        ----------
                - C : (dict) centrality for each node. Nodes are keys, centralities are values.
        
        '''
        if centrality_type in 'eigenvector':
            C = nx.algorithms.centrality.eigenvector_centrality(Gr, weight = 'weight')
        elif centrality_type in 'betweenness':
            C = nx.algorithms.centrality.betweenness_centrality(Gr)
        elif centrality_type in 'katz':
            C = nx.algorithms.centrality.katz_centrality(Gr, alpha = 0.05, beta = 1.,
                                                     weight = 'weight')
        elif centrality_type in 'current_flow':
            C = nx.algorithms.centrality.current_flow_betweenness_centrality(Gr, weight = 'weight')
        elif centrality_type in 'degree':
            C = nx.algorithms.centrality.degree_centrality(Gr)
        else:
            print('Centrality type not from among available! returning None')
            C = None
            
        # return descending order sorted centrality
        C = sorted(C.items(), key=operator.itemgetter(1))[::-1]
        return C
    
    
    def setup_nx_graph(self, edges = None, MI_thresh = None, 
                       layout = 'spring', graph_scale = 0.7):
        '''
        This code sets up the networkX graph object using provided (or stored -> self.edges) edge list.
        
        Params
        -------
                - edges : (dict, optional) if not provided uses internal attribute self.edges
                - MI_thresh : (float, optional) Minimum edge weight needed to be a member of the graph.
                - layout : (str) graph layout
                - graph_scale : (float) A number that increases / decreases separation between nodes when
                    plotted
        Returns
        --------
                - Gr : networkX graph instance
                - pos : (dict) node positions
                - weights : (list) edge weights in returned graph
        '''
        start = time()
        if MI_thresh:
            thresh = MI_thresh
        else:
            thresh = self.MI_thresh
        # build graph
        if edges:
            Gr, weights = build_graph_v2(edges, thresh)
        else:
            Gr, weights = build_graph_v2(self.edges, thresh)
        # node positions
        if layout in 'circular':
            pos = nx.circular_layout(Gr, scale=graph_scale)
        elif layout in 'spring':
            pos = nx.spring_layout(Gr, k = graph_scale, iterations=100, weight='weight')
        elif layout in 'shell':
            pos = nx.shell_layout(Gr)
        elif layout in 'spectral':
            pos = nx.spectral_layout(Gr, scale = graph_scale)
        elif layout in 'kamada':
            pos = nx.kamada_kawai_layout(Gr,scale = graph_scale)
        elif layout in 'random':
            pos = nx.random_layout(Gr)
        else:
            print('Graph layout not in available list! choosing spring')
            pos = nx.spring_layout(Gr, k = graph_scale, iterations=100, weight='weight')
            
        end = time()
        print(' ..... NX graph computed in %.2f secs ...... '%(end-start))
        return Gr, pos, weights
    
    def draw_graph(self, Gr, pos, weights, savepath = '', thresh = None, link_weight = 1.2,
                  fig_size = (40,20)):
        ''' calls draw_net '''
        # get node colors
        node_colors = color_name_map(list(Gr.nodes), self.colnames, self.child)
        
        if not thresh:
            thresh = self.MI_thresh
        start = time() 
        draw_net(Gr, pos, weights, thresh, node_colors, self.colnames, link_weight, 
             circles_to_plot = None, savepath = savepath, fig_size = fig_size)
        end = time()
        print(' ..... NX graph plotted in %.2f secs ..... '%(end-start))
        
    def save(self, Gr, savepath):
        joblib.dump(self.edges, savepath+'_edges.pkl')
        joblib.dump(Gr, savepath + '_networkX.pkl')
    
    def save_top_mi_values(self, edges, topK = None, subset_node1 = None, subset_node2 = None, savepath = ''):
        '''
        Makes a csv file with headers = (node1 , node1name, node2, node2name, weight_value) and sorts the edges
        by weight (descending order)
        
        Params
        -------
                - edges : (dict) dictionary of edges 
                - topK : (int) number of edges from the sorted edges to return as dataframe 
                - subset_node1 : (list of strings) select edges based on node 1 name
                - subset_node2 : (list of strings) select edges based on node 2 name
                - savepath : (str) path to save csv file, if empty, only returns the dataframe
        Returns
        --------
                - df : (pandas DataFrame) five columns (node1 , node1name, node2, node2name, weight_value)
        '''
        df = get_top_mi_values(edges)
        if subset_node1:
            assert type(subset_node1[0])==str, 'subset 1 has to be a string to match with node names!'
            if not subset_node2:
                df = df[ ((df['node1name'].str.startswith(subset_node1) | \
                      df['node2name'].str.startswith(subset_node2))) ]
            else:
                assert type(subset_node2[0])==str, 'subset 2 has to be a string to match with node names!'
                df = df[ (df['node1name'].str.startswith(subset_node1) | \
                      df['node2name'].str.startswith(subset_node2)) & \
                    (df['node1name'].str.startswith(subset_node2) | \
                     df['node2name'].str.startswith(subset_node2)) ]
        if topK:
            df = df.iloc[:topK, :]
        if len(savepath) > 0:
            df.to_csv(savepath, index = False)
        return df
        
    def make_graph(self, data, select_nodes = [], only_these = False, 
                   thresh = None, savepath = ''):
        self.compute_graph(data)
        if len(select_nodes) > 0:
            edges = self.select_nodes(select_nodes, only_these)
        else:
            edges = None
        if not thresh:
            thresh = self.MI_thresh
        Gr, pos, weights = self.setup_nx_graph(edges, thresh)
        self.draw_graph(Gr, pos, weights, savepath=savepath)
    
        