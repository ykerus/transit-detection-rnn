
import utils

import numpy as np
import matplotlib.pyplot as plt
import torch

from scipy.ndimage import gaussian_filter1d
from detection.detection_utils import *


def get_pts(model, flux, additional=False):
    # assuming preprocessed flux shaped as [B,N]
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = model.to(device)
    model.eval()
    flux_tens = torch.tensor(flux).type(torch.FloatTensor).to(device)
    with torch.no_grad():
        out = model(flux_tens)
        if not additional:
            return out[0].squeeze().cpu().numpy()
        return out[0].squeeze().cpu().numpy(), out[-1].squeeze().cpu().numpy()


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
    searchplus = (period < (p_est + searchdist)) & (period >= p_est)
    searchmin = (period <= p_est) & (period > (p_est - searchdist))
    # build in: stop if distance to half score in increases
    Pmin = period[searchmin][np.argmin(np.abs(score[searchmin] - half))]
    Pmax = period[searchplus][np.argmin(np.abs(score[searchplus] - half))]
    dur_est = (np.median(ntr[(period>=Pmin)&(period<=Pmax)])-1)*(Pmax-Pmin)/2
    return p_est, t0_est, dur_est, maxscore


def algorithm1(pts, num_iters=3, min_transits=3, p_min=2, p_max=None, step_mult=2, 
               smooth=True, peak_frac=2, show_steps=False, sde=False, return_steps=False):
    # folding over periods, weighing overlapping values to define a score at each time step
    def _show_step(x,y):
        plt.figure(figsize=(10,2))
        plt.plot(x,y)
        plt.show()
        
    time = np.arange(len(pts)) * utils.min2day(2)
    pts_ = gaussian_filter1d(pts.copy(), 9).reshape(1,-1) if smooth else pts.copy().reshape(1,-1)
      
    detections = {}
    steps = {"pts":[], "scores":[], "t0":[], "masks":[]}
    for i in range(num_iters):
        spectra = get_spectra(time, pts_, min_transits=min_transits, p_min=p_min, 
                              p_max=p_max, step_mult=step_mult)
        periods, scores, t0s, ntrs = spectra
        
        if sde:
            scores = (scores-np.mean(scores,1)[:,None]) / np.std(scores,1)[:,None]  # similar to BLS
        
        
        _show_step(periods, scores[0]) if show_steps else None
        candidate = find_max(periods, scores[0], t0s[0], ntrs[0], peak_frac)
        p_est, t0_est, dur_est, maxscore = candidate
        detections[maxscore] = {"period":p_est, "t0":t0_est, "duration":dur_est}
        msk = get_transit_mask(time, p_est, t0_est, dur_est, dur_mult=2)
        if return_steps:
            steps["periods"] = periods
            steps["pts"].append(pts_.copy()), steps["scores"].append(scores[0])
            steps["t0"].append(t0s), steps["masks"].append(msk)
        pts_[0,msk] = 0  # hide detected transits and run again

    if return_steps:
        return detections, steps
    return detections

# ====================================


def get_peaks(bool_array):
    # used by alg2
    if not np.any(bool_array):
        return []
    where = np.where(bool_array)[0]
    starts = np.append(0,np.where(np.diff(where, prepend=where[0])>1)[0])
    ranges = [(starts[i], starts[i+1]) for i in range(len(starts)-1)]
    indc = [where[i:j] for (i,j) in ranges] + [where[starts[-1]:]]  
    return indc
    
# def get_tc(time, peaks):
#     return np.array([np.mean(time[indc]) for indc in peaks])

def get_tc(time, peaks, pred):
    # used by alg2
    tcs = []
    for indc in peaks:
        left = np.where(np.cumsum(pred[indc])/np.sum(pred[indc])<0.5)[0]
        indx = left[-1] if len(left) > 0 else 0
        tcs.append(time[indc][indx]) 
    return np.array(tcs)

def agg_h(hiddens, peaks, agg_fn=np.mean, normalize=True):
    # used by alg2
    aggregated = np.zeros((len(peaks), hiddens.shape[-1]))
    for i, indc in enumerate(peaks):
        agg = agg_fn(hiddens[indc], axis=0)
        agg = agg / np.linalg.norm(agg) if normalize else agg
        aggregated[i] = agg
    return aggregated

def neg_mse(a, b, normalize=True):
    # used by alg2
    a_ = a / np.linalg.norm(a) if normalize else a
    b_ = b / np.linalg.norm(b) if normalize else b
    return -np.mean((a_ - b_)**2)

def dot(a, b, normalize=True):   
    # used by alg2
    a_ = a / np.linalg.norm(a) if normalize else a
    b_ = b / np.linalg.norm(b) if normalize else b
    return a_ @ b_

def match_hiddens(hiddens, sim_thresh=0.5, sim_measure="dot"):
    # used by alg2
    if sim_measure=="dot":
        aggd = np.sum((hiddens.reshape(len(hiddens),1,-1) * hiddens.reshape(1,len(hiddens),-1)),-1)
        aggd *= np.tri(*aggd.shape,k=-1).T
    return [match for match in zip(*np.where(aggd > sim_thresh))]

def agg_pred(preds, peaks, agg_fn=np.max):
    # used by alg2
    aggregated = np.zeros(len(peaks))
    for i, indc in enumerate(peaks):
        aggregated[i] = agg_fn(preds[indc])
    return aggregated

def find_candidates(matches, tcs, t_max):
    # used by alg2
    candidates = []
    match_copy = list(matches)
    for match in matches:
        match_copy.remove(match)
        match_tcs = [tcs[i] for i in match]
        if match_tcs[1] < match_tcs[0] * 2:
            continue  
        while True:
            p_expd = np.diff(match_tcs).mean()
            next_exp = match_tcs[-1] + p_expd  # expected time of next signal
            next_min, next_max = next_exp-3/24, next_exp+3/24
            if next_exp > t_max:
                candidates.append(match)
                break
            next_candidate = np.argmin(np.abs(tcs - next_exp))
            if tcs[next_candidate] < next_min or tcs[next_candidate] > next_max:
                break
            
            s = next_candidate 
            next_match = False
    
            for m in match_copy:
                
                if m[1]==s and m[0] in match:
                    next_match = True
                    break
            if next_match:
                match += (s,)
                match_tcs.append(tcs[s])
                continue
            else:
                break
    return sorted(candidates, key=len, reverse=True)

def filter_matches(matches, tcs):
    # used by alg2
    filtered = []
    for match in matches:
        match_tcs = tcs[np.array(match)]
        tcs_diffs = np.diff(match_tcs)
        period_avg = np.mean(tcs_diffs)
        if np.all(np.abs(tcs_diffs-period_avg)<30./60/24):
            filtered.append(match)
    return sorted(filtered, key=len, reverse=True)

def algorithm2(pts, reprs, num_iters=3, smooth=True, p_min=2, return_steps=False, peak_thresh=0.25):
    time = np.arange(len(pts)) * utils.min2day(2)
    pts_ = gaussian_filter1d(pts.copy(), 9) if smooth else pts.copy()
    
    peaks = get_peaks(pts_>peak_thresh) 
    if peaks is None:
        return {}
#     peak_h = agg_h(r, peaks, agg_fn=np.mean, normalize=True)  # add aggregated confidences
    peak_max = agg_pred(pts_, peaks, agg_fn=np.mean)
    peak_tc = get_tc(time, peaks, pts_)
    peak_duration = np.array([time[indc][-1]-time[indc][0] for indc in peaks])
#     match_h = match_hiddens(peak_h, sim_thresh=-99) # FIX: it matches with itself
    match_h = []
    for i in range(len(peaks)):
        match_h += [(i,j) for j in range(i+1,len(peaks)) if np.abs(peak_tc[i]-peak_tc[j])>p_min]
    
    detections = {}
    candidates = find_candidates(match_h, peak_tc, time[-1])
    if return_steps:
        steps = {"peaks":peaks, "candidates":candidates, "pts":[], "info":[], "masks":[], "best_candidates":[],
                 "tc":peak_tc}
    for i in range(num_iters):

        if len(candidates)==0:
            break
        steps["pts"].append(pts_.copy())
        best_candidate = (-1, -1)
        max_score, best_period, best_duration, best_t0 = 0, -1, -1, -1

        for c in candidates:
            try_duration = np.median(peak_duration[np.array(c)])
            try_period = np.median(np.diff(peak_tc[np.array(c)]))
            try_t0 = np.median([peak_tc[ci]-i*try_period for i,ci in enumerate(c)])
            if try_duration < 15./60/24 or try_period < p_min or try_t0 < 0:
                if return_steps:
                    steps["info"].append(f"{c} rejected: duration, P or t0 outside allowed range")
                continue
            score, n_transits = 0, 0
            tt = try_t0
            while tt < time[-1]:
                score += np.max(pts_[(time > tt-0.5*try_duration)*(time < tt+0.5*try_duration)]) # mean worse
                n_transits += 1
                tt += try_period
            score /= np.sqrt(n_transits)
            if score > max_score:
                max_score = score
                best_period = try_period
                best_t0 = try_t0
                best_duration = try_duration
                best_candidate = c
                if return_steps:
                    steps["info"].append(f"{c} new best: score = {max_score}, period = {best_period} d")
        if best_candidate != (-1,-1):
            harmonic = 2
            if best_period/2 > p_min:
                base_period = best_period
                try_period = base_period / harmonic
                while try_period > p_min:
                    try_t0 = best_t0
                    while try_t0-try_period-0.5*best_duration > 0:
                        try_t0 = try_t0-try_period

                    harmonic_score = 0
                    n_transits = 0
                    tt = try_t0
                    while tt < time[-1]:
                        harmonic_score += np.max(pts_[(time > tt-0.5*best_duration)*(time < tt+0.5*best_duration)])
                        n_transits += 1
                        tt += try_period
                    harmonic_score /= np.sqrt(n_transits)
                    if harmonic_score > max_score:
                        best_period = try_period
                        best_t0 = try_t0
                        max_score = harmonic_score

                    harmonic += 1
                    try_period = base_period/harmonic
            if return_steps:
                steps["info"].append(f"{best_candidate} harmonics evaluated: tried {harmonic-2} harmonics, "+
                                     f"new score = {max_score}, period = {best_period} d")
            detections[max_score] = {"period":best_period, "t0":best_t0, "duration":best_duration}
            msk = get_transit_mask(time, best_period, best_t0, best_duration, dur_mult=2)
            if return_steps:
                steps["masks"].append(msk.copy())
                steps["best_candidates"].append(best_candidate)
            pts_[msk] = 0
        else:
            break
    if return_steps:
        return detections, steps
    return detections
