import pandas as pd
import numpy as np
import sys, os
from fastjet import *

### converting h5 to np array, code from: https://github.com/SebastianMacaluso/TreeNiN/blob/master/code/top_reference_dataset/ReadData.py


def h5_to_npy(file, Njets, Nstart = 0):
    jets=np.array(file.select("table",start=Nstart,stop=Nstart+Njets))
    jets2=jets[:,0:800].reshape((len(jets),200,4))
    # This way I'm getting the 1st 199 constituents.
    # jets[:,800:804] is the constituent 200.
    # jets[:,804] has a label=0 for train, 1 for test, 2 for val.
    # jets[:,805] has the label sg/bg
#     print('jets2=',jets2[0])
    labels=jets[:,805:806]
#     print('labels=',labels)
    npy_jets=[]
    for i in range(len(jets2)):
        # Get the index of non-zero entries
        nonzero_entries=jets2[i][~np.all(jets2[i] == 0, axis=1)]
        npy_jets.append([nonzero_entries,0 if labels[i] == 0 else 1])
        
    # file.close()
    return npy_jets

def get_eta_phi(jets):
    # jets ordered as (px, py, pz, e)
    njets = jets.shape[0]
    nsubjets = jets.shape[1]
    phis = np.arctan2(jets[:,:,1], jets[:,:,0])
    etas = 0.5*np.log((jets[:,:,3] + jets[:,:,2])/(jets[:,:,3] - jets[:,:,2]))
    return np.append(phis.reshape(njets,nsubjets,1), etas.reshape(njets,nsubjets,1), axis=-1)

def get_dR_matrix(all_clusters, all_jet_particle_etaphis, Nsubjets, bad_jets):
    Njets = len(all_clusters)
    Nsubjets = int(Nsubjets)
    ## First, find the subjets from exclusive kT clustering
    all_subjets = []
    for ii, cs in enumerate(all_clusters):
        this_subjets = np.array([[jet.px(), jet.py(), jet.pz(), jet.e()] for jet in cs.exclusive_jets(Nsubjets)])
        inds = np.flip(np.argsort(this_subjets[:,3]))
        this_subjets = this_subjets[inds]
        all_subjets.append(this_subjets)

    ## Calculate subjet axes, do it for all jets
    all_subjets = np.array(all_subjets)
    # print("All subjets:", all_subjets[0]) ## (this would be (Njets, Nsubjets, 4)
    all_subjet_axes = get_eta_phi(all_subjets)
    # print("All subjet axes:", all_subjet_axes[0])
    all_dR_matrices = []
    
    ## For all jets, calculate the dR matrix. It would be of shape (Nsubjets, Nparticles) for each jet
    for idx in range(Njets):
        # if idx in bad_jets:
        #    continue
        # print(all_subjets[idx])
        # print(all_subjet_axes[idx])
        # this_jet_particle = np.array(npy_jets[idx][0])[:,[1,2,3,0]].reshape(1, -1, 4)
        this_jet_etaphis = all_jet_particle_etaphis[idx].reshape(-1, 2)
        this_jet_phis, this_jet_etas = this_jet_etaphis[:,0].reshape(1,-1), this_jet_etaphis[:,1].reshape(1,-1)
        # print(this_jet_phis)
        # print(this_jet_etas)
        this_jet_subjet_axes = all_subjet_axes[idx]
        this_jet_subjet_axes_phis, this_jet_subjet_axes_etas = this_jet_subjet_axes[:,0].reshape(-1,1), this_jet_subjet_axes[:,1].reshape(-1,1)
        # print(this_jet_subjet_axes_phis)
        # print(this_jet_subjet_axes_etas)
        this_dphi = this_jet_subjet_axes_phis - this_jet_phis
        this_deta = this_jet_subjet_axes_etas - this_jet_etas
        # print(this_dphi)
        # print(this_deta)
        this_dR_matrix = (this_dphi**2 + this_deta**2)**0.5
        # print(this_dR_matrix[0])
        # this_jet_particle_pts = ((this_jet_particle[:,:,0]**2 + this_jet_particle[:,:,1]**2)**0.5).reshape(-1)
        # print(this_jet_particle_pts.shape)
        all_dR_matrices.append(this_dR_matrix)
        # all_jet_particle_pts.append(this_jet_particle_pts)
        # sys.exit(1)
    # print("All dR matrix:", all_dR_matrices[0])
    # print(all_jet_particle_pts[0:5])
    return all_dR_matrices



def tauMaker(all_dR_matrices, all_jet_particle_pts):
    Njets = len(all_dR_matrices)
    Nsubjets = all_dR_matrices[0].shape[0]
    taus = np.zeros((Njets, 3))
    for idx in range(Njets):
        for ib, beta in enumerate([0.5, 1 ,2]):
            minRs = np.min(all_dR_matrices[idx]**beta, axis=0)
            taus[idx, ib] = np.sum(all_jet_particle_pts[idx] * minRs)
    return taus
            
        
def prune_dataset(npy_jets, nsubjet_min = 8):
    Njets_i = len(npy_jets)
    print("Number of pre-pruned jets: ", Njets_i)
    to_remove = []
    for ii in range(Njets_i):
        jet = npy_jets[ii][0]
        if jet.shape[0] < nsubjet_min:
            print("Jet {} has {} particles. Removing it!".format(ii, jet.shape[0]))
            to_remove.append(ii)
    return to_remove

### Working on h5 files

data_loc = "../../datasets/"
# data_files = ['train.h5', 'val.h5',  'test.h5']
data_files = ['train.h5']
Njets = None
#403000 in val, 404000 in test, 1211000 in train
Njets_per_file = 1211000
Nsubjets = list(range(1,9))
jet_def = JetDefinition(kt_algorithm, 0.4)

for filename in data_files:
    print("Filename: ", filename)
    file = pd.HDFStore(data_loc+filename)
    Njets_total = file.get_storer('table').nrows
    nfiles = int(Njets_total/Njets_per_file)
    for fid in range(nfiles):
        print("File index: ", fid)
        npy_jets = h5_to_npy(file, Njets=Njets_per_file, Nstart=fid*Njets_per_file)
        bad_jets = prune_dataset(npy_jets, nsubjet_min = 8)
        Njets = len(npy_jets) - len(bad_jets)
        print("Number of jets after pruning:", Njets)
        all_clusters = []
        all_jet_particle_pts = []
        all_jet_particle_etaphis = []
        all_labels = []
        all_jet_pts = []
        all_jet_masses = []
        for idx, (jet, label) in enumerate(npy_jets):
            if idx in bad_jets:
                continue
            # print("Jet tag: ", label)
            all_labels.append(label)
            this_particles = [PseudoJet(px, py, pz, e) for (e, px, py, pz) in jet]
            all_clusters.append(ClusterSequence(this_particles, jet_def))
            this_jet_particle_pts = ((jet[:,1]**2 + jet[:,2]**2)**0.5).reshape(-1)
            this_jet_particle_etaphis = get_eta_phi(jet[:,[1,2,3,0]].reshape(1,-1,4)).reshape(-1,2)
            this_jet_pt = (np.sum(jet[:,1])**2 + np.sum(jet[:,2])**2)**0.5
            this_jet_mass = np.abs(np.sum(jet[:,0])**2 - this_jet_pt**2 - np.sum(jet[:,3])**2)**0.5
            all_jet_particle_pts.append(this_jet_particle_pts/np.sum(this_jet_pt))
            all_jet_particle_etaphis.append(this_jet_particle_etaphis)
            all_jet_pts.append(this_jet_pt)
            all_jet_masses.append(this_jet_mass)
        all_labels = np.array(all_labels)
        all_jet_pts = np.array(all_jet_pts)
        all_jet_masses = np.array(all_jet_masses)
        np_taus = np.zeros((Njets, 3*len(Nsubjets)))
        for ii, nsubjet in enumerate(Nsubjets):
            taus = tauMaker(get_dR_matrix(all_clusters,
                                          all_jet_particle_etaphis,
                                          nsubjet,
                                          bad_jets),
                            all_jet_particle_pts)
            np_taus[:, ii*3:(ii+1)*3] = taus
        # np_taus = np.append(np_taus, all_labels.reshape(-1,1), axis = 1)
        np_taus = np.concatenate((np_taus, all_jet_pts.reshape(-1,1), all_jet_masses.reshape(-1,1), all_labels.reshape(-1,1)), axis=1)  
        print("Dataset size: ", np_taus.shape)
        np.save(data_loc + 'multi_body_tau_data/'+filename.replace(".h5", "_{}.npy".format(fid)), np_taus)
    # del npy_jets, np_taus, all_jet_particle_etaphis, all_clusters
    print("{} has been converted".format(filename))
    file.close()

