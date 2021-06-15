
import numpy as np
import matplotlib.pyplot as plt

from astropy.timeseries import BoxLeastSquares
from detection.detection_utils import *

def algorithm(time, flux, num_iters=3, min_transits=3, show_steps=False, min_p=2, freq_fac=3):
    # assuming preprocessed (detrended) flux 
    def _show_step(x,y):
        plt.figure(figsize=(10,2))
        plt.plot(x,y)
        plt.show()
        
    flux_, time_ = flux.copy(), time.copy()
    detections = {}
    for i in range(num_iters):
        model = BoxLeastSquares(time_, flux_)
        
        success = False
        while not success:
            try:
                pgram = model.autopower(np.arange(1.,13.5,.5)/24., minimum_period=min_p, 
                                        minimum_n_transit=min_transits, frequency_factor=freq_fac)
                success = True
            except ValueError:
                min_p+=1
                if min_p > time[-1]:
                    return detections
            
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