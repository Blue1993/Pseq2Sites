import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optimizer
import pickle
import os
import random
import transformers

from .Encoders import Pseq2Sites
from .helpers import prepare_prots_input

class Pseq2SitesTrainIter:
    def __init__(self, config):
        self.config = config
        
        # build model
        self.model = Pseq2Sites(self.config)
        self.modle = self.model.cuda()
        
    def train(self, train_loader, validation_loader, save_path):
        self.model_save_path = save_path
        
        # define optimizer
        self.optim = transformers.AdamW(self.model.parameters(), lr = 1e-3, weight_decay = 0.01)  
        best_eval_loss = np.inf
        
        for epoch in range(self.config["train"]["epochs"]):
            self.model.train()
            train_losses = 0.
            
            for batch in tqdm(train_loader):
                
                # prepare input
                aa_feats, prot_feats, prot_masks, binding_sites, position_ids, chain_idx = prepare_prots_input(self.config, batch)
                
                self.optim.zero_grad()
                
                # forward
                pred_BS = self.model(aa_feats, prot_feats, prot_masks, position_ids, chain_idx)
                
                # cal loss
                loss = self.get_multi_label_loss(pred_BS, binding_sites)
                loss.backward()  
                
                # backward
                self.optim.step()
                
                train_losses += loss.item()
                
            val_losses = self.eval(validation_loader)    
            
            print(f"Epochs: {epoch}, Train loss: {train_losses/len(train_loader):.3f}, Validation loss: {val_losses/len(validation_loader):.3f}")
            
            if best_eval_loss > val_losses / len(validation_loader):
                print(f"\tImprovements: {best_eval_loss - val_losses / len(validation_loader):.3f}, save {epoch} model")
                best_eval_loss = val_losses / len(validation_loader)
                
                self.save_checkpoint(
                    self.model_save_path,
                    "Pseq2Sites.pth",
                    state_dict = self.model.state_dict(),
                    optimizer = self.optim.state_dict()
                )
            
    def eval(self, loader):
        losses = 0.
        
        with torch.no_grad():
            self.model.eval()
            
            for batch in loader:
                # prepare input
                aa_feats, prot_feats, prot_masks, binding_sites, position_ids, chain_idx = prepare_prots_input(self.config, batch)
                
                # forward
                pred_BS = self.model(aa_feats, prot_feats, prot_masks, position_ids, chain_idx)
                
                # cal loss
                loss = self.get_multi_label_loss(pred_BS, binding_sites)
                losses += loss.item()

            return losses                
    
    def run_test(self, loader, best_path):
        
        #checkpoint = torch.load(config["paths"]["model_path"])
        checkpoint = torch.load(best_path + "/Pseq2Sites.pth")
        state_dict = checkpoint["state_dict"]
        self.model.load_state_dict(state_dict)
        
        predictions = list()
        
        with torch.no_grad():
            self.model.eval()
            
            for batch in loader:
                # prepare input
                aa_feats, prot_feats, prot_masks, position_ids, chain_idx = prepare_prots_input(self.config, batch, training = False)
                
                # forward
                _, pred_BS, _ = self.model(aa_feats, prot_feats, prot_masks, position_ids, chain_idx)              
                pred_BS = pred_BS * prot_masks
                predictions.extend(pred_BS.detach().tolist())
            
            return torch.nn.functional.sigmoid(torch.tensor(predictions)).numpy()
                
    def get_multi_label_loss(self, predictions, labels):
        weight = self.calculate_weights(labels)

        loss_ft = nn.BCEWithLogitsLoss(weight = weight)
        loss = loss_ft(predictions, labels)        

        return loss        
        
    def calculate_weights(self, labels):
        labels_inverse = torch.abs(labels - torch.ones(labels.size()).cuda())
        
        negative_labels = labels_inverse
        
        P = torch.sum(labels)
        N = torch.sum(negative_labels)

        P_weights = (P + N + 1) / (P + 1)
        N_weights = (P + N + 1) / (N + 1)

        weights = torch.multiply(labels, P_weights) + torch.multiply(negative_labels, N_weights)
        
        return weights 

    def save_checkpoint(self, dir, name, **kwargs):
        state = {}
        state.update(kwargs)
        filepath = os.path.join(dir, name)
        torch.save(state, filepath)          