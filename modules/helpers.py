import numpy as np
import torch

def convert_binding_site_to_labels(binding_site, max_length):
    
    '''
    Convert binding site (str) to binary label
    e.g., "3, 5, 6, 10" -> 0010110001
    '''
    
    binding_site_labels = list()
    
    for idx, bs in enumerate(binding_site):
        targets = np.array(sorted(list(set(list(map(int, bs.split(",")))))))
        one_hot_targets = np.eye(max_length)[targets]
        one_hot_targets = np.sum(one_hot_targets, axis = 0) 
        binding_site_labels.append(one_hot_targets)
    
    return binding_site_labels
    
def prepare_prots_input(config, datasets, training = True):
    
    # Get features
    prot_ids = [data[0] for data in datasets] 
    prot_feats = [data[1] for data in datasets]
    prot_seqs = [data[2] for data in datasets]

    # Collate batch data
    aa_feat, protein_feat, prots_mask, position_ids, chain_idx = collate_prots_feats(config, prot_feats, prot_seqs)
    
    # Cast to tensor
    aa_feat = torch.tensor(aa_feat, dtype = torch.float32).cuda()
    protein_feat = torch.tensor(protein_feat, dtype = torch.float32).cuda()
    prots_mask = torch.tensor(prots_mask, dtype = torch.long).cuda()
    position_ids = torch.tensor(position_ids, dtype = torch.long).cuda()
    chain_idx = torch.tensor(chain_idx, dtype = torch.long).cuda()
    
    # Convert binding sites to labels
    if training:
        prot_binding_sites = convert_binding_site_to_labels([data[3] for data in datasets], config["prots"]["max_lengths"])
        prot_binding_sites = torch.tensor(prot_binding_sites, dtype = torch.float32).cuda()
        
        return aa_feat, protein_feat, prots_mask, prot_binding_sites, position_ids, chain_idx
        
    return aa_feat, protein_feat, prots_mask, position_ids, chain_idx
    
def collate_prots_feats(config, feats, seqs):

    lengths = [len(i) for i in feats]
    max_length = config["prots"]["max_lengths"]
    hidden_dim = config["architectures"]["prots_input_dim"]
    input_mask = [[1] * length for length in lengths]
    
    position_id, chain_idx = list(), list()
    aa_feat = np.zeros((len(feats), max_length, hidden_dim))
    protein_feat = np.zeros((len(feats), max_length, hidden_dim))
    
    for idx, line in enumerate(feats):
        seq_length = line.shape[0]
        aa_feat[idx,:seq_length,:] = line # amino aicd-level embeddings
        
        pad_mask = [0] * (max_length - seq_length)
        input_mask[idx].extend(pad_mask)
        
        ### position index
        position_id.append([i for i in range(max_length)])
        
        ### protein-level embeddings
        seq_list = seqs[idx].split(',')
        start_seq, end_seq = 0, 0
        tmp_chain_idx = list()
        
        for jdx, chain_seq in enumerate(seq_list):
            end_seq += len(chain_seq)
            protein_feat[idx, start_seq:end_seq, :] = np.sum(line[start_seq:end_seq, :])
            start_seq = end_seq
            
            ### chain index
            for i in range(len(chain_seq)):
                tmp_chain_idx.append(jdx + 1)
        
        for i in range(seq_length, max_length):
            tmp_chain_idx.append(0)
        
        chain_idx.append(tmp_chain_idx)

    return aa_feat, protein_feat, input_mask, position_id, chain_idx

def convert_bs(pred_binding_sites):
    
    prediction_row, prediction_cols = np.where((pred_binding_sites>0.8) == True)
    
    final_results, check_row, check_col = list(), set(), list()
    
    for pred_r, pred_c in zip(prediction_row, prediction_cols):
        if len(check_row) == 0 and pred_r not in check_row:
            check_row.add(pred_r)
            check_col.append(pred_c)
            
        elif pred_r in check_row:
            check_col.append(pred_c)
            
        elif pred_r not in check_row:    
            final_results.append(",".join(list(map(str, check_col))))
            check_col = list()
            check_row.add(pred_r)
            check_col.append(pred_c)
            
    final_results.append(",".join(list(map(str, check_col))))
    
    return final_results

def get_results(binding_sites, pred_binding_sites, sequences):
    from sklearn.metrics import confusion_matrix
    
    T_TP, T_TN, T_FP, T_FN = 0, 0, 0, 0
    
    for bs, bps, seq in zip(binding_sites, pred_binding_sites, sequences):
        seq_len = len([i for i in seq if i != ","])
        index = [str(i) for i in range(seq_len)]
        
        positive_label = set(bs.split(","))
        negative_label = set(index) - set(positive_label)
        positive_pred = set(bps.split(","))
        negative_pred = set(index) - set(positive_pred)
        
        TP = len(positive_pred & positive_label)
        TN = len(negative_pred & negative_label)
        FP = len(positive_pred & negative_label)
        FN = len(negative_pred & positive_label) 
        
        T_TP += TP
        T_TN += TN
        T_FP += FP
        T_FN += FN
        
    precision = T_TP / (T_TP + T_FP)
    recall = T_TP / (T_TP + T_FN)
    specificity = T_TN / (T_TN + T_FP) 
    G_mean = np.sqrt(specificity * recall)
    ACC = (T_TP + T_TN) / (T_TP + T_TN + T_FP + T_FN)
    F1_score = (2 * precision * recall) / (precision + recall)
    F2_score = (5 * precision * recall) / (4 * precision + recall)
    
    return np.round(precision,2), np.round(recall,2), np.round(specificity, 2), np.round(ACC, 2), np.round(G_mean, 2), np.round(F1_score, 2), np.round(F2_score, 2)

