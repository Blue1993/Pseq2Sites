import yaml
import os
import logging
import sys
import pandas as pd
import numpy as np
import pickle
import argparse

from modules.utils import load_cfg
from modules.data import PocketDataset, Dataloader
from modules.TrainIters import Pseq2SitesTrainIter
from modules.helpers import *

def main():

    """ Define argparse """
    parser = argparse.ArgumentParser(
            description = "Pseq2Sites predicts binding site based on protein sequence information"
    )
    
    parser.add_argument("--config", "-c", required = True, type = input_check, 
                    help = "The file contains information on the protein sequences to predict binding sites. \
                    (refer to the examples.inp file for a more detailed file format)")

    parser.add_argument("--labels", "-l", required = True, type = bool,
                        help = "labels is True: Binding site information is use for evaluation performance; \
                                labels is False: When protein' binding site information is unknwon; \
                                e.g., -t True" 
                )

    args = parser.parse_args()
    
    config = load_cfg(args.config)
    
    print("1. Load data ...")
    """ Load protein info """
    if args.labels:
        with open(config["paths"]["prot_feats"], "rb") as f:
            IDs, sequences, binding_sites, protein_feats = pickle.load(f)    

    else:
        with open(config["paths"]["prot_feats"], "rb") as f:
            IDs, sequences, protein_feats = pickle.load(f) 
    
    print("2. Make dataset ...")
    dataset = PocketDataset(IDs, protein_feats, sequences)
    loader = Dataloader(dataset, batch_size = config["train"]["batch_size"], shuffle = False)

    print("3. Binding sites prediction ...")
    trainiter = Pseq2SitesTrainIter(config)
    predicted_binding_sites = trainiter.run_test(loader, config["paths"]["model_path"])
    
    print("4. Write predicted binding sites ...")
    if args.labels:
        fwrite(config["paths"]["result_path"], IDs, pred_binding_sites = predicted_binding_sites, binding_sites = binding_sites)
    else:
        fwrite(config["paths"]["result_path"], IDs, pred_binding_sites = predicted_binding_sites)

def fwrite(path, IDs, pred_binding_sites = None, binding_sites = None):
    fw = open(path, "w")
    
    pred_binding_sites = convert_bs(pred_binding_sites)
    
    if binding_sites[0] != None:
        fw.write("PDB\tBS\tPred_BS\n")
        for id_, bs, pbs in zip(IDs, binding_sites, pred_binding_sites):
            fw.write(f"{id_}\t{bs}\t{pbs}\n")
        fw.close()    
    else:
        fw.write("PDB\tPred_BS\n")
        for id_, pbs in zip(IDs, pred_binding_sites):
            fw.write(f"{id_}\t{pbs}\n")
        fw.close()
     
def input_check(path):

    path = os.path.abspath(path)
    if not os.path.exists(path):
        raise IOError('%s does not exist.' %path)
    return path  
    
if __name__ == "__main__":
    main()  
