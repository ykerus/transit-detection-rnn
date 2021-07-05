import numpy as np
import utils
from scipy.ndimage import gaussian_filter1d
from detection import rnn_detection as rnndet


def monotransit_detection(pts, conf=None, peak_thresh=0.5, agg_fn=np.multiply, score_fn=np.max, smooth=True):
    # assuming uniform time
    time = np.arange(len(pts)) * utils.min2day(2)
    conf = np.ones_like(pts) if conf is None else conf
    
    pts_ = gaussian_filter1d(pts.copy(), 9) if smooth else pts.copy()
    # conf_ = gaussian_filter1d(conf.copy(), 9) if smooth else conf
    comb_ = gaussian_filter1d(agg_fn(conf, pts), 9) if smooth else agg_fn(conf, pts)
    
    tr_indc = rnndet.get_peaks(pts_>=peak_thresh)
    
    conf_scores = [score_fn(comb_[indc]) for indc in tr_indc]
    standard_scores = [score_fn(pts_[indc]) for indc in tr_indc]
    tc = rnndet.get_tc(time, tr_indc, pts_)
    
    conf_detections, standard_detections = {}, {}
    for i, indc in enumerate(tr_indc):
        dur_est = time[indc[-1]] - time[indc[0]]
        standard_detections[standard_scores[i]] = {"t0":tc[i], "duration":dur_est}
        # if conf_scores[i] >= peak_thresh:
        conf_detections[conf_scores[i]] = {"t0":tc[i], "duration":dur_est} 
    return standard_detections, conf_detections
    
 