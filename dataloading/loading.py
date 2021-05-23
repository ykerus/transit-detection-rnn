
import numpy as np
import torch
import data_processing as dp

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


def get_loaders_fn(train_path, valid_path, train_batch=128, valid_batch=1000, test_path=None, 
                   mode=1, nanmode=2, scale_median=0, standardize=1,
                   incl_centr=False):
    # preprocesses function for own simulated data

    train_path = train_path if isinstance(train_path, list) else [train_path]
    valid_path = valid_path if isinstance(valid_path, list) else [valid_path]
    test_path = test_path if isinstance(test_path, list) else [test_path]
    
    # train data
    split = combine_data([dp.load_data(path) for path in train_path])
    split["flux"], (mean, std), addnl = dp.preprocess(split["flux"], split["sigma"], mode, nanmode,
                                        None, None, scale_median, True, standardize,  
                                        centr=[split["mom_col"], split["mom_row"]] if incl_centr else None,
                                        centr_mean=None, centr_std=None)
    del split["mom_col"], split["mom_row"]
    (centr, centr_mean, centr_std) = addnl
    additional = centr if incl_centr else None
    
    train_loader = DataLoader(dp.Data(split["flux"], split["mask"], split["transit"], split["rdepth"],
                                      additional=additional),
                              batch_size=train_batch, shuffle=True)  
    del split
    
    # validation data
    split = combine_data([dp.load_data(path) for path in valid_path])
    split["flux"], _ , addnl = dp.preprocess(split["flux"], split["sigma"], mode, nanmode,
                                     mean, std, scale_median, True, standardize,
                                     centr=[split["mom_col"], split["mom_row"]] if incl_centr else None,
                                     centr_mean=centr_mean, centr_std=centr_std)
    del split["mom_col"], split["mom_row"]
    (centr, _, _) = addnl
    additional = centr if incl_centr else None

    valid_loader = DataLoader(dp.Data(split["flux"], split["mask"], split["transit"], split["rdepth"],
                                      additional=additional),
                              batch_size=valid_batch, shuffle=False)
    if test_path[0] is None:
        return train_loader, valid_loader, None
    
    # test data
    split = combine_data([dp.load_data(path) for path in test_path])
    split["flux"], _, addnl = dp.preprocess(split["flux"], split["sigma"], mode, nanmode,
                                     mean, std, scale_median, True, standardize,
                                     centr=[split["mom_col"], split["mom_row"]] if incl_centr else None,
                                     _mean=centr_mean, centr_std=centr_std)
    del split["mom_col"], split["mom_row"]
    (centr, _, _) = addnl
    additional = centr if incl_centr else None
    test_loader = DataLoader(dp.Data(split["flux"], split["mask"], split["transit"], split["rdepth"],
                                     additional=additional),
                              batch_size=valid_batch, shuffle=False)
    return train_loader, valid_loader, test_loader