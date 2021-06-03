
import pickle
import numpy as np
import torch
from torch.utils.data import DataLoader
from dataloading import data_processing as dp


class Data(torch.utils.data.Dataset):
    def __init__(self, flux, mask, transit, rdepth, additional=None):
        # any data array in additional should be same size as flux
        if additional is not None:
            additional = additional if isinstance(additional, list) else [additional]
            self.flux = torch.tensor(np.stack([flux] + additional, axis=2)).type(torch.FloatTensor)  # [B, T, F]
        else:
            self.flux = torch.tensor(flux).type(torch.FloatTensor).view(len(flux), -1)  # [B, T]
        self.mask = torch.tensor(mask).type(torch.FloatTensor).view(len(flux), -1)  # [B, T]
        self.transit = torch.tensor(transit).type(torch.FloatTensor).view(-1)  # [B,]
        self.rdepth = torch.tensor(rdepth).type(torch.FloatTensor).view(len(flux), -1)  # [B, T]
        self.n_samples = len(flux[:, 0])

    def __getitem__(self, key):
        return self.flux[key], self.mask[key], self.transit[key], self.rdepth[key]

    def __len__(self):
        return self.n_samples
    
    
class ContrastData(torch.utils.data.Dataset):
    def __init__(self, flux1, mask1, hmask1, transit1, rdepth1, flux2, mask2, hmask2, transit2, rdepth2, positive):
        # any data array in additional should be same size as flux
        self.flux1 = torch.tensor(flux1).type(torch.FloatTensor).view(len(flux1), -1)  # [B, T]
        self.flux2 = torch.tensor(flux2).type(torch.FloatTensor).view(len(flux2), -1)  # [B, T]
        
        self.mask1 = torch.tensor(mask1).type(torch.FloatTensor).view(len(flux1), -1)  # [B, T]
        self.mask2 = torch.tensor(mask2).type(torch.FloatTensor).view(len(flux2), -1)  # [B, T]
        
        self.hmask1 = torch.tensor(hmask1).type(torch.FloatTensor).view(len(flux1), -1)  # [B, T]
        self.hmask2 = torch.tensor(hmask2).type(torch.FloatTensor).view(len(flux2), -1)  # [B, T]
        
        self.transit1 = torch.tensor(transit1).type(torch.FloatTensor).view(-1)  # [B,]
        self.transit2 = torch.tensor(transit2).type(torch.FloatTensor).view(-1)  # [B,]
        
        self.rdepth1 = torch.tensor(rdepth1).type(torch.FloatTensor).view(len(flux1), -1)  # [B, T]
        self.rdepth2 = torch.tensor(rdepth2).type(torch.FloatTensor).view(len(flux2), -1)  # [B, T]
        
        self.positive = torch.tensor(positive).type(torch.FloatTensor).view(-1)  # [B,]
        
        self.n_samples = len(flux1[:, 0])

    def __getitem__(self, key):
        out1 = (self.flux1[key], self.mask1[key], self.hmask1[key], self.transit1[key], self.rdepth1[key])
        out2 = (self.flux2[key], self.mask2[key], self.hmask2[key], self.transit2[key], self.rdepth2[key])
        return out1, out2, self.positive[key]

    def __len__(self):
        return self.n_samples


def load_data(load_path, unpack=None):
    if load_path.split("/")[-1].startswith("."):
        return [None]*6 if unpack else None
    with open(load_path, "rb") as f:
        batch = pickle.load(f)
    if unpack is None or unpack == False:
        return batch
    unpack = [unpack] if isinstance(unpack, str) else unpack
    return [batch[key] for key in unpack]


def combine_data(data_parts):
    combined = data_parts[0]
    for part in data_parts[1:]:
        for d in part:
            combined[d] = np.concatenate((combined[d], part[d]), axis=0)
    return combined


def separate_trues(bool_array):  
    if not np.any(bool_array):
        return []
    where = np.where(bool_array)[0]
    starts = np.append(0,np.where(np.diff(where, prepend=where[0])>1)[0])
    ranges = [(starts[i], starts[i+1]) for i in range(len(starts)-1)]
    indc = [where[i:j] for (i,j) in ranges] + [where[starts[-1]:]]  
    return indc


def insert_gaps_fn(flux, mask, inplace=True, min_hours=2, max_hours=10):
    num_gaps = np.random.choice([0,1,2], p=[0.5, 0.35, 0.15])
    
    flux_ = flux if inplace else flux.copy()
    tr_indc = separate_trues(mask.astype(bool))
    edge_free = 6 * 30  # points away from edge to place gaps
    gap_spacing = 30  # min space between gaps
    
    success = False
    while not success:
        spacing_success = True
        gap_ranges = []
        gap_msk = np.zeros(len(flux)).astype(bool)
        for gap in range(num_gaps):
            size = int(np.exp(np.random.uniform(np.log(min_hours), np.log(max_hours)))*30)
            i = np.random.choice(np.arange(int(edge_free+size/2), len(flux)-int(edge_free+size/2)))
            min_i, max_i = int(i-size/2), int(i+size/2)
            for rng in gap_ranges:
                if (max_i < rng[0] and rng[0]-max_i < gap_spacing)\
                or (min_i > rng[1] and min_i - rng[1] < gap_spacing)\
                or (max_i > rng[0] and max_i < rng[1])\
                or (min_i > rng[0] and min_i < rng[1]):
                    spacing_success = False
            gap_ranges.append((min_i,max_i))
            gap_msk[min_i:max_i] = True
        gap_msk[np.random.choice([True, False], p=[0.02, 0.98], size=len(gap_msk))] = True
        gap_msk[0], gap_msk[-1] = False, False
        success = np.all([gap_msk[indc].mean()<0.5 for indc in tr_indc]) * spacing_success

    flux_[gap_msk] = np.nan
    return flux_


def get_loaders_fn(train_path, valid_path, train_batch=128, valid_batch=1000, test_path=None, 
                   mode=1, nanmode=2, scale_median=0, standardize=1,
                   incl_centr=False, insert_gaps=False):
    # preprocesses function for own simulated data

    train_path = train_path if isinstance(train_path, list) else [train_path]
    valid_path = valid_path if isinstance(valid_path, list) else [valid_path]
    test_path = test_path if isinstance(test_path, list) else [test_path]
    
    # train data
    split = combine_data([load_data(path) for path in train_path])
    if insert_gaps:
        for i in range(len(split["flux"])):
            split["flux"][i] = insert_gaps_fn(split["flux"][i], split["mask"][i])
        
    split["flux"], (mean, std), addnl = dp.preprocess(split["flux"], split["sigma"], mode, nanmode,
                                        None, None, scale_median, True, standardize,  
                                        centr=[split["mom_col"], split["mom_row"]] if incl_centr else None,
                                        centr_mean=None, centr_std=None)
    if "mom_col" in split:
        del split["mom_col"], split["mom_row"]
    (centr, centr_mean, centr_std) = addnl
    additional = centr if incl_centr else None
    
    train_loader = DataLoader(Data(split["flux"], split["mask"], split["transit"], split["rdepth"],
                                      additional=additional),
                              batch_size=train_batch, shuffle=True)  
    del split
    
    # validation data
    split = combine_data([load_data(path) for path in valid_path])
    if insert_gaps:
        for i in range(len(split["flux"])):
            split["flux"][i] = insert_gaps_fn(split["flux"][i], split["mask"][i])
    split["flux"], _ , addnl = dp.preprocess(split["flux"], split["sigma"], mode, nanmode,
                                     mean, std, scale_median, True, standardize,
                                     centr=[split["mom_col"], split["mom_row"]] if incl_centr else None,
                                     centr_mean=centr_mean, centr_std=centr_std)
    if "mom_col" in split:
        del split["mom_col"], split["mom_row"]
    (centr, _, _) = addnl
    additional = centr if incl_centr else None

    valid_loader = DataLoader(Data(split["flux"], split["mask"], split["transit"], split["rdepth"],
                                      additional=additional),
                              batch_size=valid_batch, shuffle=False)
    if test_path[0] is None:
        return train_loader, valid_loader, None
    
    # test data
    split = combine_data([load_data(path) for path in test_path])
    if insert_gaps:
        for i in range(len(split["flux"])):
            split["flux"][i] = insert_gaps_fn(split["flux"][i], split["mask"][i])
    split["flux"], _, addnl = dp.preprocess(split["flux"], split["sigma"], mode, nanmode,
                                     mean, std, scale_median, True, standardize,
                                     centr=[split["mom_col"], split["mom_row"]] if incl_centr else None,
                                     centr_mean=centr_mean, centr_std=centr_std)
    if "mom_col" in split:
        del split["mom_col"], split["mom_row"]
    (centr, _, _) = addnl
    additional = centr if incl_centr else None
    test_loader = DataLoader(Data(split["flux"], split["mask"], split["transit"], split["rdepth"],
                                     additional=additional),
                              batch_size=valid_batch, shuffle=False)
    return train_loader, valid_loader, test_loader


def get_contrast_loaders_fn(train_path, valid_path, train_batch=128, valid_batch=1000, test_path=None, 
                            mode=1, nanmode=2, scale_median=0, standardize=1):
    
    # train data
    split = load_data(train_path)
    fl_out, (mean, std), _ = dp.preprocess(np.concatenate([split["flux1"],split["flux2"]]),
                                          np.repeat(split["sigma"],2), mode, nanmode,
                                          None, None, scale_median, True, standardize)
    split["flux1"], split["flux2"] = np.split(fl_out, 2)

    
    train_loader = DataLoader(ContrastData(split["flux1"], split["mask1"], split["hmask1"],
                                           split["transit1"], split["rdepth1"], split["flux2"], 
                                           split["mask2"], split["hmask2"], split["transit2"],
                                           split["rdepth2"], split["positive"]),
                              batch_size=train_batch, shuffle=True)
    del split
    
    # validation data
    split = load_data(valid_path)
    fl_out, _, _ = dp.preprocess(np.concatenate([split["flux1"],split["flux2"]]),
                                          np.repeat(split["sigma"],2), mode, nanmode,
                                          mean, std, scale_median, True, standardize)
    split["flux1"], split["flux2"] = np.split(fl_out, 2)

    valid_loader = DataLoader(ContrastData(split["flux1"], split["mask1"], split["hmask1"],
                                           split["transit1"], split["rdepth1"], split["flux2"], 
                                           split["mask2"], split["hmask2"], split["transit2"],
                                           split["rdepth2"], split["positive"]),
                              batch_size=valid_batch, shuffle=False)
    del split
    if test_path[0] is None:
        return train_loader, valid_loader, None
    
    # test data
    split = load_data(test_path)
    fl_out, _, _ = dp.preprocess(np.concatenate([split["flux1"],split["flux2"]]),
                                          np.repeat(split["sigma"],2), mode, nanmode,
                                          mean, std, scale_median, True, standardize)
    split["flux1"], split["flux2"] = np.split(fl_out, 2)

    test_loader = DataLoader(ContrastData(split["flux1"], split["mask1"], split["hmask1"],
                                           split["transit1"], split["rdepth1"], split["flux2"], 
                                           split["mask2"], split["hmask2"], split["transit2"],
                                           split["rdepth2"], split["positive"]),
                              batch_size=valid_batch, shuffle=False)
    return train_loader, valid_loader, test_loader