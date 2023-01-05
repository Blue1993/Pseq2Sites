import pandas as pd
import numpy as np
import argparse
import re
import torch
import pickle
import numpy as np
import os
from pathlib import Path
from transformers import T5EncoderModel, T5Tokenizer

def main():
    
    parser = argparse.ArgumentParser(
        description = "Generate input data for Pseq2Sites. Sequence embeddings are extracted from the sequence using a pre-trained model,\
                       and in the case of a multi-chain protein, embeddings are extracted after splitting by chain."
        )
    parser.add_argument("--input", "-i", required = True, type = str,
                        help ="The file contains information such as protein ids and sequences; \
                               Enter the path of preprocessing.py output; \
                               e.g., -i ./data/COACH420_data.csv"
                )
    parser.add_argument("--output", "-o", required = True, type = str,
                        help = "The file contains infomration for training and test; \
                                It is saved in pickle file format and has the following order: \
                                IDs, chains, seqs, binding sites, features; \
                                Enter the path to save; \
                                e.g., -o ./data/COACH420_features.pkl"  
                )  
    parser.add_argument("--training", "-t", required = True, type = bool,
                        help = "training is True: Binding site information is added when generating data for training; \
                                training is False: Binding site information is not added when generating data for testing; \
                                e.g., -t True" 
                )
    
    args = parser.parse_args()
    
    input_abspath = os.path.abspath(args.input)
    if not os.path.isfile(input_abspath):
        raise IOError(f"Plase check input file path; {input_abspath} does not exist")

    output_abspath = os.path.abspath(args.output)
    
    if not os.path.isdir(os.path.abspath(os.path.dirname(args.output))):
        raise IOError(f"Plase check output dir path; {output_abspath} does not exist")
    
    max_len = 1500
    
    print("1. Load data ...")
    prots_df = pd.read_csv(args.input, sep = "\t")
    
    if args.training:
        IDs, seqs, binding_sites = prots_df.iloc[:,0].values, prots_df.iloc[:,1].values, prots_df.iloc[:,2].values
    else:
        IDs, seqs = prots_df.iloc[:,0].values, prots_df.iloc[:,1].values

    print("2. Load tokenizer and pretrained model")
    tokenizer = T5Tokenizer.from_pretrained("Rostlab/prot_t5_xl_uniref50", do_lower_case=False )
    prots_model = T5EncoderModel.from_pretrained("Rostlab/prot_t5_xl_uniref50")
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    prots_model = prots_model.to(device)
    prots_model = prots_model.eval()
    
    batch_seq_list, prots_feat_list = list(), list()
    
    for seq in seqs[16000:]:
        # split to chain seqs
        seq_list = seq.split(",")
        tmp_seq_embeddings = list()
        
        # get chain embeddings
        for se in seq_list:
            batch_seq_list.append(se)
            seqs_example = [re.sub(r"[UZOB]", "X", seq) for seq in batch_seq_list]
            seqs_example = [" ".join(list(seq)) for seq in seqs_example]

            ids = tokenizer.batch_encode_plus(seqs_example, add_special_tokens = True, pad_to_max_length = True)
            input_ids = torch.tensor(ids['input_ids']).to(device)
            attention_mask = torch.tensor(ids['attention_mask']).to(device)  
            
            with torch.no_grad():  
                embedding = prots_model(input_ids = input_ids, attention_mask = attention_mask)[0]
                embedding = embedding.cpu().numpy()
                seq_len = (attention_mask[0] == 1).sum()

                # prot_t5 do not use start token
                seq_emd = embedding[0][:seq_len-1]

            batch_seq_list = list()
            tmp_seq_embeddings.extend(list(seq_emd))
            
        prots_feat_list.append(np.array(tmp_seq_embeddings))
    
    if args.training:
        with open(output_abspath, "wb") as f:        
            pickle.dump((IDs, seqs, binding_sites, prots_feat_list), f)     
    else:
        with open(output_abspath, "wb") as f:        
            pickle.dump((IDs, seqs, prots_feat_list), f)   

if __name__ == "__main__":
    main()