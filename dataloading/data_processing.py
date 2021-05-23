
import numpy as np
import warnings
from scipy import interpolate
from wotan import flatten
import utils

def lin_interp(flux, flat_window=None, pos_offs=1e5, t_step=utils.min2day(2),
               inplace=True):
    # assumes uniformly spaced measurements
    if flux.ndim == 1:
        nan = np.isnan(flux)
        if ~np.any(nan):
            return flux
        time = np.arange(len(flux)) * t_step
        if flat_window is not None:
            f_, trend = flatten(time, flux+pos_offs, method="median",
                                window_length=flat_window, return_trend=1)
            trend -= pos_offs  # offset necessary if flux if zero centered
        else:
            trend = flux
        f = interpolate.interp1d(time[~nan], trend[~nan])
        flux_new = flux if inplace else flux.copy()
        flux_new[nan] = f(time[nan])
        return flux_new
    else:
        flux_interp = flux if inplace else flux.copy()
        for i in range(len(flux)):
            flux_interp[i] = lin_interp(flux[i], flat_window)
        return flux_interp

    
def uniform_time(time, data, cadence=utils.min2day(2), offset=None):
    offset = cadence/2 if offset is None else offset
    data = [data] if not isinstance(data, list) else data 
    
    t_new = [time[0]]
    d_new = [[d[0]] for d in data]
    
    for i, t in enumerate(time[1:], 1):
        prev = t_new[-1]
        while t - prev > cadence + offset:
            prev = prev + cadence
            t_new.append(prev)
            for j in range(len(data)):
                d_new[j].append(np.nan)
        t_new.append(t)
        for j in range(len(data)):
            d_new[j].append(data[j][i])
    return np.array(t_new), (np.array(d) for d in d_new)


def separate_trues(bool_array):  
    if not np.any(bool_array):
        return []
    where = np.where(bool_array)[0]
    starts = np.append(0,np.where(np.diff(where, prepend=where[0])>1)[0])
    ranges = [(starts[i], starts[i+1]) for i in range(len(starts)-1)]
    indc = [where[i:j] for (i,j) in ranges] + [where[starts[-1]:]]  
    return indc


def make_flat(time, flux, window=0.5):
    return flatten(time, flux, method="median", window_length=window)


def get_outliers(flux, lower=5, upper=5, sigma=None):
    if sigma is not None:
        lower = upper = sigma
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        outliers = sigma_clip(flux, sigma_lower=lower, sigma_upper=upper).mask
    return outliers


def preprocess(flux, scaling=None, mode=0, nanmode=0, mean=None, std=None,
               scale_median=False, center_zero=True, standardize=True,
               centr=None, centr_mean=None, centr_std=None,
               window=utils.min2day(60)): # window for interpolation
    # modes: 0:nothing, 1:scale, 2:scale+diff, 3:diff+scale
    # nanmodes: 0:nothing, 1:zero-fill, 2:lin_interp
    

    # scaling is array of individual scaling factors (len(scaling)==len(flux))
    flux_median = np.nanmedian(flux, axis=1)[:, None]
    flux_ = flux / flux_median if scale_median else flux.copy()
    centr_ = None
    if centr is not None:
        centr_ = [c - np.nanmedian(c, axis=1)[:, None] if center_zero else c.copy() for c in centr]
        centr_ = [c / flux_median for c in centr_] if scale_median else centr_

    flux_ = flux_ - 1 if center_zero else flux_  # center around zero

    # mode 0: nothing
    nan = np.isnan(flux_)
    if mode == 1:  # scale
        if nanmode == 1:
            flux_[nan] = 0
            if centr is not None:
                centr_[0][nan], centr_[1][nan] = 0, 0
        elif nanmode == 2:
            flux_ = lin_interp(flux_, window)
            if centr is not None:
                centr_[0] = lin_interp(centr_[0], window)
                centr_[1] = lin_interp(centr_[1], window)
        flux_ /= scaling[:, None]

    elif mode == 2:  # scale + diff
        if nanmode == 1:
            flux_[nan] = 0
            if centr is not None:
                centr_[0][nan], centr_[1][nan] = 0, 0
        elif nanmode == 2:
            flux_ = lin_interp(flux_, window)
            if centr is not None:
                centr_[0] = lin_interp(centr_[0], window)
                centr_[1] = lin_interp(centr_[1], window)
        flux_ /= scaling[:, None]
        flux_ = np.diff(flux_, prepend=flux_[:, 0][:, None])

    elif mode == 3:  # diff + scale
        scaling = np.nanstd(np.diff(flux_, prepend=flux_[:, 0][:, None]), axis=1)
        if nanmode > 0:
            if nanmode == 1:
                flux_[nan] = 0
                if centr is not None:
                    centr_[0][nan], centr_[1][nan] = 0, 0
            elif nanmode == 2:
                flux_ = lin_interp(flux_, window)
                if centr is not None:
                    centr_[0] = lin_interp(centr_[0], window)
                    centr_[1] = lin_interp(centr_[1], window)
        flux_ = np.diff(flux_, prepend=flux_[:, 0][:, None])
        flux_ /= scaling[:, None]
        
    if mode > 0 and centr is not None:
        centr_[0] /= scaling[:, None]
        centr_[1] /= scaling[:, None]

    mean = np.nanmean(flux_) if mean is None else mean
    std = np.nanstd(flux_) if std is None else std

    centr_mean = [np.nanmean(c) for c in centr_] if (centr_mean is None and centr is not None) else centr_mean
    centr_std = [np.nanstd(c) for c in centr_] if (centr_std is None and centr is not None) else centr_std

    if standardize:
        flux_ = (flux_ - mean) / std  # standardize
        if centr is not None:
            centr_[0] = (centr_[0] - centr_mean[0]) / centr_std[0]
            centr_[1] = (centr_[1] - centr_mean[1]) / centr_std[1]
    return flux_, (mean, std), (centr_, centr_mean, centr_std)