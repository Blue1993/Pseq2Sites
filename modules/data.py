from torch.utils.data import Dataset, DataLoader

def Dataloader(dataset, batch_size, shuffle = True, drop_last = False): 
    
    data_loader = DataLoader(
                    dataset,
                    batch_size = batch_size,
                    shuffle = shuffle,
                    drop_last = drop_last,
                    collate_fn = lambda x:x,
                    pin_memory = True
                )
                
    return data_loader
    
class PocketDataset:
    def __init__(self, IDs, feats, seqs, binding_sites = None):
        self.IDs = IDs
        self.feats = feats
        self.seqs = seqs
        self.binding_sites = binding_sites
 
    def __len__(self):
        return len(self.IDs)
    
    def __getitem__(self, idx):
        if self.binding_sites is not None:
            return self.IDs[idx], self.feats[idx], self.seqs[idx], self.binding_sites[idx]
        else:
            return self.IDs[idx], self.feats[idx], self.seqs[idx] 
