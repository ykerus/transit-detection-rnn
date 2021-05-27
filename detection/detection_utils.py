
import numpy as np

def get_transit_mask(time, period, t0, duration, dur_mult=1):  
    msk = np.zeros_like(time, dtype=bool)
    tt = t0
    while tt < time[-1]:
        msk[(time >= tt - (dur_mult*duration/2.)) & (time <= tt + (dur_mult*duration/2.))] = True
        tt += period
    return msk