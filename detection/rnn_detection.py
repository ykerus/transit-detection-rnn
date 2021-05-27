
import utils

import numpy as np
import matplotlib.pyplot as plt
import torch

from scipy.ndimage import gaussian_filter1d
from detection.detection_utils import *


def get_pts(model, flux):
    # assuming preprocessed flux shaped as [B,N]
    flux_tens = torch.tensor(flux).type(torch.FloatTensor)
    with torch.no_grad():
        pts = model(flux_tens)[0]
    return pts.numpy()


def fold_multi(time, data, period):
    # used by alg1
    indc = np.arange(len(time))
    period_n = int(period / (2/60/24))
    if period_n >= len(indc):
        return time.reshape(1,-1), data
    
    h = int(np.ceil(len(indc)/period_n))
    pad = (h*period_n) % len(indc)
    shape = [data.shape[0], -1, period_n]
    
    padding = np.zeros((data.shape[0],pad))*np.nan
    time_fold = np.append(time, padding[0]).reshape(*shape[1:])
    data_fold = np.append(data, padding, 1).reshape(*shape)
    
    n_tr = np.concatenate((np.zeros(period_n-pad, dtype=int)+h, -np.ones(pad, dtype=int)+h))
    
    return time_fold, data_fold, n_tr 


def get_spectra(time, pts_multi, min_transits=3, p_min=1, p_max=None, step_mult=1):
    # used by alg1
    # assuming uniform time, with time[0] = 0
    periods, scores, t0s, ntrs = [], [], [], []

    if p_max is None:
        p_max = time[-1] / (min_transits-1) if min_transits > 1 else time[-1]
    
    # the following could be improved
    steps = [time[time<2.5][::2*step_mult],
             time[(time>=2.5)&(time<4)][::3*step_mult],
             time[(time>=4)&(time<6)][::5*step_mult],
             time[(time>=6)&(time<9)][::8*step_mult],
             time[(time>=9)][::13*step_mult]]
    steps = np.concatenate(steps)
    for p_try in steps:
        if p_try > p_min and p_try < p_max:
            periods.append(p_try)
            tfold, ffold, n_tr = fold_multi(time, pts_multi, p_try)
            score = np.nansum(ffold, axis=1) / (n_tr[None,:]**(1/2))#np.sqrt(n_tr)
            eval_tr = n_tr >= min_transits
            max_score = np.argmax(score[:,eval_tr], 1)
            scores.append(score[np.arange(score.shape[0]),max_score])
            t0s.append(tfold[[0]*len(max_score),max_score])
            ntrs.append(n_tr[max_score])
    return np.array(periods), np.vstack(scores).T, np.vstack(t0s).T, np.vstack(ntrs).T


def find_max(period, score, t0, ntr, peak_frac=2):
    # used by alg1
    maxscore = np.max(score)  # assumes 0 baseline
    argmax = np.argmax(score)
    half = maxscore / peak_frac
    p_est, t0_est = period[argmax], t0[argmax]
    searchdist = 0.02 * p_est
    searchplus = (period < (p_est + searchdist)) & (period > p_est)
    searchmin = (period < p_est) & (period > (p_est - searchdist))
    # build in: stop if distance to half score in increases
    Pmin = period[searchmin][np.argmin(np.abs(score[searchmin] - half))]
    Pmax = period[searchplus][np.argmin(np.abs(score[searchplus] - half))]
    dur_est = (np.median(ntr[(period>=Pmin)&(period<=Pmax)])-1)*(Pmax-Pmin)/2
    return p_est, t0_est, dur_est, maxscore


def algorithm1(pts, num_iters=3, min_transits=3, p_min=2, p_max=None, step_mult=2, 
               smooth=True, peak_frac=2, show_steps=False):
    def _show_step(x,y):
        plt.figure(figsize=(10,2))
        plt.plot(x,y)
        plt.show()
        
    time = np.arange(len(pts)) * utils.min2day(2)
    pts_ = gaussian_filter1d(pts.copy(), 9).reshape(1,-1) if smooth else pts.copy().reshape(1,-1)
      
    detections = {}
   
    for i in range(num_iters):
        spectra = get_spectra(time, pts_, min_transits=min_transits, p_min=p_min, 
                              p_max=p_max, step_mult=step_mult)
        periods, scores, t0s, ntrs = spectra
        sdes = (scores-np.mean(scores,1)[:,None]) / np.std(scores,1)[:,None]  # similar to BLS
        _show_step(periods, sdes[0]) if show_steps else None
        candidate = find_max(periods, sdes[0], t0s[0], ntrs[0], peak_frac)
        p_est, t0_est, dur_est, maxscore = candidate
        detections[maxscore] = {"period":p_est, "t0":t0_est, "duration":dur_est}
        msk = get_transit_mask(time, p_est, t0_est, dur_est, dur_mult=2)
        pts_[0,msk] = 0  # hide detected transits and run again
    return detections