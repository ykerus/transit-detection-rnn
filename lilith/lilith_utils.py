import os
import numpy as np
import pickle
import utils


def get_pl_data(path_to_gt, sampleids):
    pl_data = np.loadtxt(path_to_gt +"/tsop301_planet_data.txt")
    
    pl_sampleids = pl_data[:,0].astype("int")
    pl_num = pl_data[:,1].astype("int")
    pl_orb_period = pl_data[:,2]
    pl_t0 = pl_data[:,3]
    pl_ror = pl_data[:,4]
    pl_a = pl_data[:,7]  # [r_star]
    pl_duration = pl_data[:,8] / 24.  # [days]
    pl_depth = pl_data[:,9] * 1e-6 # mid-transit
    
    pl_unique_ids = np.unique(pl_sampleids)
    pl_dic = {s_id:{} for s_id in sampleids}
    for s_id in pl_unique_ids:
        if s_id not in sampleids:
            print("WARNING: sampleid in planet data not found in sampleids")
        indc = np.where(pl_sampleids == s_id)[0]
        for i in indc:
            pl_dic[s_id][pl_num[i]] = {"orb_period":pl_orb_period[i], "t0":pl_t0[i], 
                                       "ror":pl_ror[i], "duration":pl_duration[i],
                                       "a":pl_a[i], "depth":pl_depth[i]}
    return pl_dic, pl_unique_ids


def get_eb_data(path_to_gt, sampleids, backeb=False):
    fname = "tsop301_backeb_data.txt" if backeb else "tsop301_eb_data.txt"
    eb_data = np.loadtxt(path_to_gt + "/" + fname)
    
    eb_sampleids = eb_data[:,0].astype("int")
    eb_orb_period = eb_data[:,2]
    eb_t0 = eb_data[:,3]
    eb_primary_depth = eb_data[:,7] * 1e-6
    eb_secondary_depth = eb_data[:,8] * 1e-6
    
    eb_unique_ids = np.unique(eb_sampleids)
    eb_dic = {s_id:{} for s_id in sampleids}
    for s_id in eb_unique_ids:
        if s_id not in sampleids:
            print(f"WARNING: sampleid {s_id} in (B)EB data not found in sampleids")
        i = np.where(eb_sampleids==s_id)[0][0]
        eb_dic[s_id] = {"orb_period":eb_orb_period[i], "t0":eb_t0[i], 
                        "depth_1":eb_primary_depth[i], "depth_2":eb_secondary_depth[i]}
    return eb_dic, eb_unique_ids


def get_single_mask(time, tt, orb_period, duration):
    pl_mask = np.zeros_like(time, dtype=bool)
    # adjust t0 to lie in correct range if possible
    tmin, tmax = np.nanmin(time), np.nanmax(time)
    while tt >= tmin:  # sometimes t0 is way too high for the given light curve
        tt -= orb_period
    while tt < tmin:  # sometimes t0 is too low for the given light curve
        tt += orb_period
    while tt <= tmax:
        pl_mask[(time>=tt-duration/2) * (time<=tt+duration/2)] = True
        tt += orb_period
    return pl_mask
    
    
def get_transit_masks(time, pl_meta, lc_std=None, incl_params=False):
    pl_masks = np.zeros((len(pl_meta), len(time)), dtype=bool)
    rel_depth = np.zeros(len(time)) if incl_params else None
    tr_n = np.zeros(len(time)) if incl_params else None
    for n in pl_meta:
        pl_mask = get_single_mask(time, pl_meta[n]["t0"], pl_meta[n]["orb_period"], pl_meta[n]["duration"])
        pl_masks[n-1] = pl_mask
        if incl_params:
            rel_depth[pl_mask] = pl_meta[n]["depth"] / lc_std
            tr_n[pl_mask] = int(pl_meta[n]["duration"] / utils.min2day(2))
    mask = pl_masks.any(0)
    overlap = pl_masks.sum(0) > 1
    if incl_params:
        tr_n[overlap] = rel_depth[overlap] = -1
        return pl_masks, mask, overlap, rel_depth, tr_n
    return pl_masks, mask, overlap


def get_bash_file(path_to_fits):
    # assuming bash (.sh) file is located in same dir as fits
    for fname in os.listdir(path_to_fits):
        if fname.endswith(".sh"):
            return fname
        
def sampleid_from_fname(fname):
    # from fits fname
    if fname.endswith(".fits.gz"):
        return int(fname.split("-")[2])
    return None


def sampleids_from_fits(path_to_fits):
    # from dir containing fits files
    sampleids = []
    for fname in os.listdir(path_to_fits):
        sampleid = sampleid_from_fname(fname)
        if sampleid is not None:
            sampleids.append(sampleid)
    return np.array(sampleids).astype(int)


def sampleids_from_curl(path_to_curl):
    # from dir containing curl file
    sampleids = []
    with open(path_to_curl+"/"+get_bash_file(path_to_curl), "r") as f:
        lines = f.read().splitlines()
    for line in lines:
        if line.endswith(".fits.gz"):
            sampleid = sampleid_from_fname(line.split("/")[-1])
            sampleids.append(sampleid)
    return np.array(sampleids).astype(int)


def combine_data(data_parts):
    combined = data_parts[0]
    for part in data_parts[1:]:
        for d in part:
            combined[d] = np.concatenate((combined[d], part[d]), axis=0)
    return combined
