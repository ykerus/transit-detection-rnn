import os
import pickle
import numpy as np
from tqdm import tqdm

import utils
from simulations import lightcurve_simulation as lcsim


def generate_nn_dataset(dir_name, size, pl_fracs=None, t_step=utils.min2day(2), snr_range=(3,80), 
                        rdepth_range=(.25,5), N_points=1500, dur_range=(0,utils.hour2day(14)),
                        period_range=(2,100), base_dir="data/nn/sim", lower_snr=True):
    pl_fracs = [0.5, 0.35, 0.15] if pl_fracs is None else pl_fracs
    
    store_path = base_dir + '/' + dir_name
    utils.make_dir(base_dir)  # if it doesn't exist yet
    
    N_samples = [int(frac * size) for frac in pl_fracs[:-1]]
    N_samples += [size-sum(N_samples)]

    flux_all = np.zeros((size, N_points))
    mask_all = np.zeros((size, N_points), dtype=bool)  # transit mask
    rdepth_all = np.zeros((size, N_points))  # relative depth (/sigma)
    sigma_all = np.zeros(size)  # (estimated) time-indep noise
    
    samples_done = [0 for i in range(len(N_samples))]
    planets = np.where(np.array(N_samples) > 0)[0][0]
    
    pbar = tqdm(range(size))
    for i in pbar:
        try:  
            success = False
            while not success:
                lc = lcsim.get_lightcurve(num_planets=planets, t_step=t_step, t_max=N_points*t_step,
                                          rdepth_range=rdepth_range, min_transits=1, snr_range=snr_range, 
                                          period_range=period_range, dur_range=dur_range, lower_snr=lower_snr)
                _, flux, masks, params = lc
                success = (flux is not None)
            flux_all[i] = flux
            mask_all[i] = np.any(masks, axis=0)
            depths = [msk*params["planets"][pl_i]["pl_ror"]**2 for pl_i, msk in enumerate(masks)]
            rdepth_all[i] = np.zeros(len(flux)) + np.sum(depths,0)/params["sigma"]
            sigma_all[i] = params["sigma"]
            
            samples_done[planets] += 1
            if samples_done[planets] == N_samples[planets]:
                planets += 1
        except:
            pbar.close()
            raise
                
    dset = {"flux":flux_all, "mask":mask_all, "transit":mask_all.any(1), 
            "rdepth":rdepth_all, "sigma":sigma_all}
    
    with open(store_path, "wb") as f:
        pickle.dump(dset, f)


def generate_eval_dataset(dir_name, size, pl_fracs=None, t_step=utils.min2day(2), snr_range=(3,80), 
                          rdepth_range=(.25,5), t_max=27.4, dur_range=(0,utils.hour2day(14)),
                          period_range=(2,100), base_dir="data/eval/sim", 
                          min_tr=1, max_tr=50, batch_save=250, lower_snr=False):
    pl_fracs = [0.5, 0.5] if pl_fracs is None else pl_fracs
    N_points = int(t_max / utils.min2day(2)) 
    
    store_path = base_dir + '/' + dir_name
    utils.make_dir(store_path)  # if it doesn't exist yet
    
    N_samples = [int(frac * size) for frac in pl_fracs[:-1]]
    N_samples += [size-sum(N_samples)]

    def _empty_arrays():
        arrays = (np.zeros((batch_save, N_points)), np.zeros((batch_save, N_points), dtype=bool),
                  np.zeros(batch_save), np.zeros(batch_save), {})
        return arrays
    
    flux_b, mask_b, sigma_b, sampleid_b, meta_b = _empty_arrays()
    
    samples_done = [0 for i in range(len(N_samples))]
    planets = np.where(np.array(N_samples) > 0)[0][0]
    
    bi = 0
    pbar = tqdm(range(size))
    for i in pbar:
        try:  
            success = False
            while not success:
                lc = lcsim.get_lightcurve(num_planets=planets, t_step=t_step, t_max=N_points*t_step,
                                          rdepth_range=rdepth_range, min_transits=min_tr, snr_range=snr_range, 
                                          period_range=period_range, dur_range=dur_range, lower_snr=lower_snr)
                _, flux, masks, params = lc
                success = (flux is not None)
                if success and planets>0:
                    for planet in range(planets):
                        n_tr = params["planets"][planet]["pl_transits"]
                        if n_tr < min_tr or n_tr > max_tr:
                            success = False
                    
            flux_b[bi] = flux
            mask_b[bi] = np.any(masks, axis=0)
            sigma_b[bi] = params["sigma"]
            sampleid_b[bi] = i
            meta_b[i] = params
            
            bi += 1
            samples_done[planets] += 1
            if samples_done[planets] == N_samples[planets]:
                planets += 1
                
            if bi==batch_save:
                dbatch = {"flux":flux_b, "mask":mask_b, "transit":mask_b.any(1), 
                          "sigma":sigma_b, "sampleid":sampleid_b, "meta":meta_b}
                start, end = f"{(i+1-batch_save)}".zfill(5), f"{i}".zfill(5)
                with open(store_path+f"/{start}-{end}", "wb") as f:
                    pickle.dump(dbatch, f)
                bi = 0 
                flux_b, mask_b, sigma_b, sampleid_b, meta_b = _empty_arrays()
                        
        except:
            pbar.close()
            raise
                    