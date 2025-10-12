import torch
from functools import partial
from torch.utils.data import DataLoader
from torch import nn

def collate_fn(batch):
    """Custom collator that flattens the 12-sequence groups written by
    ``prepare_ampdfm_dataset.py``.  After collation each field has the shape
    expected by the models: *all sequences in the batch* are stacked along the
    first dimension.

    Returned shapes
    ----------------
    input_ids      (B×G, L)
    attention_mask (B×G, L)
    cond_vec       (B×G, 4)
    where *G* is the group size used during dataset packing (12 for PepDFM).
    """

    flat_input_ids = []
    flat_attention = []
    flat_cond_vec  = []

    for item in batch:  # each *item* is one Arrow record containing 12 seqs
        flat_input_ids.extend(item['input_ids'])
        flat_attention.extend(item['attention_mask'])

        # ``cond_vec`` is per-sequence; fall back to zeros if missing.
        if 'cond_vec' in item:
            flat_cond_vec.extend(item['cond_vec'])
        else:
            flat_cond_vec.extend([[0, 0, 0, 0]] * len(item['input_ids']))

    input_ids = torch.tensor(flat_input_ids)
    attention_mask = torch.tensor(flat_attention)
    cond_vec = torch.tensor(flat_cond_vec, dtype=torch.float32)

    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'cond_vec': cond_vec,
    }

class CustomDataModule(nn.Module):
    def __init__(self, train_dataset, val_dataset, test_dataset, collate_fn=collate_fn):
        super().__init__()
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.collate_fn = collate_fn

    def train_dataloader(self):
        return DataLoader(self.train_dataset, 
                          collate_fn=partial(self.collate_fn),
                          num_workers=8,
                          pin_memory=True,
                          shuffle=True)
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset, 
                          collate_fn=partial(self.collate_fn),
                          num_workers=8,
                          pin_memory=True,
                          shuffle=False)
  
    def test_dataloader(self):
        return DataLoader(self.test_dataset, 
                          collate_fn=partial(self.collate_fn),
                          num_workers=8,
                          pin_memory=True,
                          shuffle=False)