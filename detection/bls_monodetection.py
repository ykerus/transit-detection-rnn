import numpy as np
from scipy.ndimage import gaussian_filter1d
from detection import rnn_detection as rnndet

def search(tau, t, x, y, ivar, apodize=1.0):
    # https://github.com/dfm/peerless/blob/main/peerless/_search.pyx
    i0 = 0
    nt = t.shape[0]
    nx = x.shape[0]
    out_depth = np.zeros(nt)
    out_depth_ivar = np.zeros(nt)
    
    for n in range(nt):
        tmn = t[n] - tau/2
        tmx = tmn + tau/2
        depth = 0.0
        depth_ivar = 0.0

        # Find the starting coordinate.
        for i in range(i0, nx):
            if x[i] >= tmn:
                i0 = i
                break

        # Compute the model online.
        for i in range(i0, nx):
            if x[i] >= tmx:
                break

            # Any missing points.
            if np.isnan(y[i]):
                depth = 0.0
                depth_ivar = 0.0
                break

            # Compute the depth online.
            depth += ivar[i] * y[i]
            depth_ivar += ivar[i]

        if depth_ivar > 0.0:
            depth /= -depth_ivar

        # Save these results.
        out_depth[n] = depth
        out_depth_ivar[n] = depth_ivar

    s2n = out_depth * np.sqrt(out_depth_ivar)
    if apodize > 0.0:
        s2n /= 1.0 + np.exp(-(t - (t[0] + apodize * tau)) / tau)
        s2n /= 1.0 + np.exp((t - (t[-1] - apodize * tau)) / tau)

    return out_depth, out_depth_ivar, s2n

def get_time_indc(lctime, durations, grid_frac):
    indc = np.zeros((len(durations), len(lctime)), dtype=int)        
    for i, tau in enumerate(durations):
        time_ = np.arange(lctime.min(), lctime.max(), grid_frac * tau)
        diff = np.abs(lctime.reshape(-1,1) - time_)
        indc[i] = np.argmin(diff, 1)
    return indc

    
def monotransit_detection(lctime, flat, unc, durations=None, grid_frac=0.25, time_indc=None, return_indc=False, 
                         smooth=True, peak_thresh=10, score_fn=np.max, return_s2n=False):
    # time_indc help transfering detections back to larger time grid, and compare multiple durations
    durations = np.arange(1.,13.5,2)/24. if durations is None else durations
    time_indc = get_time_indc(lctime, durations, grid_frac) if time_indc is None else time_indc
    
    s2n = np.zeros_like(time_indc, dtype=float)
    depth = np.zeros_like(time_indc, dtype=float)
    depth_ivar = np.zeros_like(time_indc, dtype=float)
    
    for i, tau in enumerate(durations):
        time_ = np.arange(lctime.min(), lctime.max(), grid_frac * tau)
        d, d_ivar, s = search(tau, time_, lctime, flat-1, np.ones(len(flux))*1/unc**2) 
        s2n[i] = s[time_indc[i]]
        depth[i] = d[time_indc[i]]
        depth_ivar[i] = d_ivar[time_indc[i]]

    argmax = np.argmax(s2n, 0)
    x_indc = np.arange(len(lctime), dtype=int)
    
    s2n = s2n[argmax, x_indc]
    s2n = gaussian_filter1d(s2n.copy(), 9) if smooth else s2n
    depth = depth[argmax, x_indc]
    depth_ivar = depth_ivar[argmax, x_indc]
    
    m = depth_ivar > 0.0 # TODO: check nans effect
    noise = np.nan + np.zeros_like(s2n)
    _, noise[m] = flatten(lctime[m], np.abs(s2n[m]), method="median", window_length=6, return_trend=1)
    
    sde = (s2n - s2n.mean()) / s2n.std()
    
    s2n_detections = {} # choose low peak_thresh so other methods will be completes
    dfm_detections = {}
    sde_detections = {} # sde is not completely right here, but include for comparison
    
    tr_indc = rnndet.get_peaks(s2n>=peak_thresh)
    tc = rnndet.get_tc(lctime, tr_indc, s2n)
    s2n_scores = [score_fn(s2n[indc]) for indc in tr_indc]
    dfm_scores = [score_fn(s2n[indc]/noise[indc]) for indc in tr_indc]
    sde_scores = [score_fn(sde[indc]) for indc in tr_indc]
    
    for i, indc in enumerate(tr_indc):
        dur_est = lctime[indc[-1]] - lctime[indc[0]]
        s2n_detections[s2n_scores[i]] = {"t0":tc[i], "duration":dur_est}
        dfm_detections[dfm_scores[i]] = {"t0":tc[i], "duration":dur_est}
        sde_detections[sde_scores[i]] = {"t0":tc[i], "duration":dur_est}
        
    if return_indc and not return_s2n:
        return s2n_detections, dfm_detections, sde_detections, time_indc
    elif return_s2n and not return_indc:
        return s2n_detections, dfm_detections, sde_detections, 
    elif return_indc and return_s2n:
        return s2n_detections, dfm_detections, sde_detections, time_indc, s2n
    return s2n_detections, dfm_detections, sde_detections