import os
import pickle
import numpy as np
from tqdm import tqdm

import utils
from simulations import lightcurve_simulation as lcsim


def generate_nn_dataset(dir_name, size, pl_fracs=None, t_step=utils.min2day(2), snr_range=(3,80), 
                        rdepth_range=(.25,5), N_points=1500, dur_range=(0,utils.hour2day(13)),
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
        
        
def generate_contrast_dataset(dir_name, size, t_step=utils.min2day(2), snr_range=(3,80), 
                        rdepth_range=(.25,5), N_points=1500, dur_range=(utils.hour2day(1),utils.hour2day(13)),
                        period_range=(2,100), base_dir="data/nn/sim", lower_snr=False, hsize=20):
    # hsize is the number of points to be evaluated for representation learning
    # should be same for each sample, otherwise difficult vectorization
    
    pl1_fracs = [0.5, 0.2, 0.3]  # together: [0.5, 0.35, 0.15]
    pl2_fracs = [[0.4, 0.1], [0.04, 0.16], [0.06, 0.24]]
    
    store_path = base_dir + '/' + dir_name
    utils.make_dir(base_dir)  # if it doesn't exist yet
    
    N1_samples = [int(frac * size) for frac in pl1_fracs[:-1]]
    N1_samples += [size-sum(N1_samples)]
    
    N2_samples = [[int(fr[0]*size),int(fr[1]*size)] for fr in pl2_fracs]

    flux1_all = np.zeros((size, N_points))
    mask1_all = np.zeros((size, N_points), dtype=bool)  # transit mask
    hmask1_all = np.zeros((size, N_points), dtype=bool)  # repr mask
    rdepth1_all = np.zeros((size, N_points))  # relative depth (/sigma)
    
    flux2_all = np.zeros((size, N_points))
    mask2_all = np.zeros((size, N_points), dtype=bool)  # transit mask
    hmask2_all = np.zeros((size, N_points), dtype=bool)  # repr mask
    rdepth2_all = np.zeros((size, N_points))  # relative depth (/sigma)
    
    sigma_all = np.zeros(size)  # (estimated) time-indep noise
    positive = np.zeros(size, dtype=bool)
    
    samples_done1 = [0 for i in range(len(N1_samples))]
    samples_done2 = [[0, 0] for i in range(len(N1_samples))]
    planets1 = np.where(np.array(N1_samples) > 0)[0][0]
    planets2 = 0
        
    pbar = tqdm(range(size))
    for i in pbar:
        try:  
            if planets1==0 and planets2==0:
                positive[i] = True
            elif not ((planets1==0 and planets2!=0) or (planets1!=0 and planets2==0)):
                positive[i] = np.random.choice([True, False])
            
            success = False
            while not success:
          
                lc1 = lcsim.get_lightcurve(num_planets=planets1, t_step=t_step, t_max=N_points*t_step,
                                          rdepth_range=rdepth_range, min_transits=1, snr_range=snr_range, 
                                          period_range=period_range, dur_range=dur_range, lower_snr=lower_snr)
                _, flux1, masks1, params1 = lc1
                success1 = (flux1 is not None)
                if success1:
                    st_params = lcsim.get_stellar_params(rot_sigma=params1["rot_sigma"], 
                                                         rot_period=params1["rot_period"], 
                                                         gran_period=params1["gran_period"], 
                                                         gran_sigma=params1["gran_sigma"])
                    if planets1>0 and positive[i]:
                        pl_params = lcsim.get_planetary_params(pl_period=params1["planets"][0]["pl_period"], 
                                                               pl_ror=params1["planets"][0]["pl_ror"], 
                                                               pl_inc=params1["planets"][0]["pl_inc"],
                                                               pl_ecc=params1["planets"][0]["pl_ecc"], 
                                                               pl_w=params1["planets"][0]["pl_w"],
                                                               st_u=params1["planets"][0]["st_u"], 
                                                               st_M=params1["planets"][0]["st_M"], 
                                                               st_R=params1["planets"][0]["st_R"])
                    elif planets1>0 and planets2>0 and (not positive[i]):
                        pl_params = lcsim.get_planetary_params(pl_w=params1["planets"][0]["pl_w"],
                                                               st_u=params1["planets"][0]["st_u"], 
                                                               st_M=params1["planets"][0]["st_M"], 
                                                               st_R=params1["planets"][0]["st_R"])
                    else:
                        pl_params = None

                    lc2 = lcsim.get_lightcurve(num_planets=planets2, t_step=t_step, t_max=N_points*t_step,
                                          rdepth_range=rdepth_range, min_transits=1, snr_range=snr_range, 
                                          period_range=period_range, dur_range=dur_range, lower_snr=lower_snr,
                                          sigma=params1["sigma"], pl_params=pl_params, st_params=st_params)
                    _, flux2, masks2, params2 = lc2
                    success2 = (flux2 is not None)
                    success = success1 * success2
             
            flux1_all[i] = flux1
            mask1 = np.any(masks1, axis=0)
            mask1_all[i] = mask1
            hmask1 = np.zeros_like(mask1, dtype=bool)
            if np.any(mask1):
                hmask1[np.random.choice(np.where(masks1[0])[0], size=hsize, replace=False)] = True
            else:
                hmask1[np.random.choice(np.arange(len(mask1)), size=hsize, replace=False)] = True
            hmask1_all[i] = hmask1
            depths1 = [msk*params1["planets"][pl_i]["pl_ror"]**2 for pl_i, msk in enumerate(masks1)]
            rdepth1_all[i] = np.zeros(len(flux1)) + np.sum(depths1,0)/params1["sigma"]
            
            flux2_all[i] = flux2
            mask2 = np.any(masks2, axis=0)
            mask2_all[i] = mask2
            hmask2 = np.zeros_like(mask2, dtype=bool)
            if np.any(mask2):
                hmask2[np.random.choice(np.where(masks2[0])[0], size=hsize, replace=False)] = True
            else:
                hmask2[np.random.choice(np.arange(len(mask2)), size=hsize, replace=False)] = True
            hmask2_all[i] = hmask2
            depths2 = [msk*params2["planets"][pl_i]["pl_ror"]**2 for pl_i, msk in enumerate(masks2)]
            rdepth2_all[i] = np.zeros(len(flux2)) + np.sum(depths2,0)/params2["sigma"]
            
            sigma_all[i] = params1["sigma"]
            
            samples_done1[planets1] += 1
            samples_done2[planets1][planets2] += 1
            if samples_done2[planets1][planets2] == N2_samples[planets1][planets2]:
                planets2 += 1
            if samples_done1[planets1] == N1_samples[planets1]:
                planets1 += 1
                planets2 = 0
                
        except:
            pbar.close()
            raise
                
    dset = {"flux1":flux1_all, "mask1":mask1_all, "hmask1":hmask1_all, "transit1":mask1_all.any(1), 
            "rdepth1":rdepth1_all, "rdepth2":rdepth2_all, "sigma":sigma_all, "positive":positive,
            "flux2":flux2_all, "mask2":mask2_all, "hmask2":hmask2_all, "transit2":mask2_all.any(1)}
    
    with open(store_path, "wb") as f:
        pickle.dump(dset, f)


def generate_eval_dataset(dir_name, size, pl_fracs=None, t_step=utils.min2day(2), snr_range=(3,80), 
                          rdepth_range=(.25,5), t_max=27.4, dur_range=(0,utils.hour2day(13)),
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
                    