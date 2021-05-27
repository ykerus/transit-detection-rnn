
import numpy as np
import matplotlib.pyplot as plt

from astropy.timeseries import BoxLeastSquares
from detection.detection_utils import *

def algorithm(time, flux, num_iters=3, show_steps=False):
    # assuming preprocessed (detrended) flux 
    def _show_step(x,y):
        plt.figure(figsize=(10,2))
        plt.plot(x,y)
        plt.show()
        
    flux_, time_ = flux.copy(), time.copy()
    detections = {}
    for i in range(num_iters):
        model = BoxLeastSquares(time_, flux_)
        pgram = model.autopower(np.arange(1.,13.5,.5)/24., minimum_period=2, minimum_n_transit=3, frequency_factor=3)
        sde = (pgram.power - np.mean(pgram.power)) / np.std(pgram.power)
        _show_step(pgram.period,sde) if show_steps else None
        maxscore = np.max(sde)
        argmax = np.argmax(sde)
        p_est = pgram.period[argmax]
        dur_est = pgram.duration[argmax]
        t0_est = pgram.transit_time[argmax]
        detections[maxscore] = {"period":p_est, "t0":t0_est, "duration":dur_est}
        msk = get_transit_mask(time_, p_est, t0_est, dur_est, dur_mult=2)
        flux_, time_ = flux_[~msk], time_[~msk]
    return detections