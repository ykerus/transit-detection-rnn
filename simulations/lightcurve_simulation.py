import utils
import numpy as np
import celerite2
import batman
import astropy.constants as const


def get_stellar_params(rot_sigma=None, rot_period=None, gran_period=None, gran_sigma=None):
    
    if rot_sigma is None:
        # affects fluctuation amplitude
        rot_sigma = np.exp(np.random.uniform(np.log(0.0003), np.log(0.005)))
    if rot_period is None:
        # inspiration from: https://arxiv.org/pdf/2007.03079.pdf Fig.5
        rot_period = np.abs(np.random.normal(5, 2))+1
        
    # from: https://gallery.exoplanet.codes/en/latest/tutorials/stellar-variability/
    rot_Q0 = np.random.lognormal(0, 2)  # from *
    rot_dQ = np.random.lognormal(0, 2)  # from *
    rot_f = np.random.uniform(0.1, 1)  # from *
        
    if gran_period is None:
        # inspiration from: https://arxiv.org/pdf/1408.0817.pdf Fig.8 
        gran_freq = np.random.lognormal(4.5,1)*1e-6  # Hz
        gran_period = utils.min2day(1. / gran_freq / 60) # days
    if gran_sigma is None:
        # inspiration from: https://arxiv.org/pdf/1408.0817.pdf Fig.9
        gran_sigma = np.random.lognormal(np.log(2e-6 * gran_freq**-0.61), 0.1)
    
    params = {"rot_sigma":rot_sigma, "rot_period":rot_period, 
              "rot_Q0":rot_Q0, "rot_dQ":rot_dQ, "rot_f":rot_f, 
              "gran_period":gran_period, "gran_sigma":gran_sigma}
    return params


def get_stellar_variability(time, params=None):
    if params is None:
        params = get_stellar_params()
        
    # https://celerite2.readthedocs.io/en/latest/api/python/
    k_rot = celerite2.terms.RotationTerm(sigma=params["rot_sigma"], period=params["rot_period"],
                                          Q0=params["rot_Q0"], dQ=params["rot_dQ"], f=params["rot_f"])
    k_gran = celerite2.terms.SHOTerm(rho=params["gran_period"], sigma=params["gran_sigma"], 
                                     Q=1/np.sqrt(2))
    kernel = k_rot + k_gran
    gp = celerite2.GaussianProcess(kernel=kernel, t=time, mean=1.0, diag=np.zeros(len(time)))
    gp.compute(time)
    flux = gp.sample()
    
    return flux, params


def get_photon_noise(time, sigma=None):
    if sigma is None:
        sigma = np.exp(np.random.uniform(np.log(0.0005), np.log(0.003)))
    noise = np.random.normal(0, sigma, len(time))
    return noise, sigma


def get_planetary_params(pl_period=None, pl_ror=None, pl_inc=None, pl_ecc=None, pl_w=None,
                         st_u=None, st_M=None, st_R=None,
                         period_range=(2,100)):
    # excluding t0 because it depends on the time and data gaps
    
    if pl_period is None:
        per_min, per_max = period_range
        pl_period = np.exp(np.random.uniform(np.log(per_min), np.log(per_max)))
    if pl_ror is None:  # planet radius (Rpl/R*)
        pl_ror = np.exp(np.random.uniform(np.log(0.02), np.log(0.15)))
    if pl_inc is None:  # orbital inclination (in degrees)
        pl_inc = 90. 
    if pl_ecc is None:  # eccentricity
        pl_ecc = np.random.beta(a=0.867, b=3.03)  # https://arxiv.org/pdf/1306.4982.pdf
    if pl_w is None: # longitude of periastron (in degrees)
        pl_w = np.random.uniform(0, 360) # 90 is shortest duration, 270 is longest
    
    if st_u is None:  # limb darkening coeffs
        q1, q2 = np.random.uniform(0, 1, 2)  # https://arxiv.org/pdf/1308.0009.pdf
        st_u = [2 * np.sqrt(q1) * q2, np.sqrt(q1) * (1 - 2 * q2)]
    
    # the following is just to get a reasonable and slightly random relation
    # between period P and semi-major axis a
    if st_M is None:
        st_M = np.abs(np.random.normal(0.9, 0.25))+0.1
        st_R = np.abs(np.random.normal(st_M**(1/3.)-0.1, 0.2))+0.1  # overwrite st_R
    if st_R is None:
        st_R = np.abs(np.random.normal(M_star**(1/3.)-0.1, 0.2))+0.1
    M_ = st_M * const.M_sun.value  # kg
    R_ = st_R * const.R_sun.value  # m
    P_ = pl_period * 24 * 60 * 60  # s
    a_ = ((P_ / (2*np.pi))**2 * M_ * const.G.value)**(1/3.)  # m
    pl_a = a_ / R_  # semi-major axis of orbit

    params = {"pl_period":pl_period, "pl_a":pl_a, "pl_ror":pl_ror, "pl_inc":pl_inc, 
              "pl_ecc":pl_ecc, "pl_w":pl_w, "st_u":st_u, "st_M":st_M, "st_R":st_R}
    return params


def get_transits(time, params=None, min_transits=2, mask=None, max_attempts=10):
    # if mask is given, place transits s.t. they are non-overlapping
    if params is None:
        params = get_planetary_params()
    
    p = batman.TransitParams()
    p.per = params["pl_period"]
    p.rp = params["pl_ror"]
    p.a = params["pl_a"]
    p.inc = params["pl_inc"]
    p.ecc = params["pl_ecc"]
    p.w = params["pl_w"]
    p.limb_dark = "quadratic"
    p.u = params["st_u"]
    
    p.t0 = 0.5
    time_single = np.arange(0, 1, utils.min2day(1))
    m = batman.TransitModel(p, time_single)
    flux_single = m.light_curve(p)
    indc = np.where(flux_single<1)[0]
    params["pl_duration"] = time_single[indc[-1]] - time_single[indc[0]]
    
    min_t0 = time[0] + params["pl_duration"]  # at least half the duration from edge
    # make sure t0 is really the first transit
    max_t0 = min(time[-1]-params["pl_duration"]-(min_transits-1)*p.per, p.per-0.5*params["pl_duration"])
    t0_error = min_t0 > max_t0
    
    m = batman.TransitModel(p, time)
    
    attempts = 0
    overlap = True
    while overlap and attempts < max_attempts and not t0_error:
        p.t0 = np.random.uniform(min_t0, max_t0)
        flux = m.light_curve(p)
        overlap = False if mask is None else np.any((mask*(flux < 1)) > 0)
        attempts += 1
    if overlap or t0_error:
        return None, params
    # else... non-overlapping transits
    params["pl_t0"] = p.t0
    return flux, params


def _add_planet(time, rdepth_range, snr_range, mask, max_snr_attempts, pl_given, pl_params,
                sigma, min_transits, period_range, pl_i, info, dur_range, lower_snr):
    # used by get_lightcurve, placed here to split up the code
    snr, snr_attempt, snr_success = 0, 0, False
    while not snr_success and snr_attempt < max_snr_attempts:
        # pl_ror only used if pl_params is not provided, or with multiple planets
        if lower_snr:
            mn, mx = rdepth_range[0]**2, rdepth_range[1]**2  # squared for more bias towards lower snr
        else: 
            mn, mx = rdepth_range[0], rdepth_range[1]
        rdepth = (np.exp(np.random.uniform(np.log(mn), np.log(mx)))-mn)/(mx-mn)
        rdepth = rdepth * (rdepth_range[1]-rdepth_range[0]) + rdepth_range[0]
        pl_ror = np.sqrt(rdepth * sigma)
        
        if pl_i==0 and pl_given is None:
            pl_params = get_planetary_params(period_range=period_range, pl_ror=pl_ror)        
        elif pl_i > 0:
            pl_params = get_planetary_params(period_range=period_range, st_u=pl_params["st_u"], 
                                 st_M=pl_params["st_M"], st_R=pl_params["st_R"], pl_ror=pl_ror)
        else:
            pl_params = pl_given.copy()

        transits, pl_params = get_transits(time, params=pl_params, min_transits=min_transits, mask=mask)
        if transits is None or pl_params["pl_duration"]<dur_range[0] or pl_params["pl_duration"]>=dur_range[1]:
            snr_attempt += 1
            continue
            
        n = int(pl_params["pl_duration"] / (time[1]-time[0]))  # num data points covered by transit
        snr = pl_params["pl_ror"]**2 / sigma * np.sqrt(n)  # in reality lower, time-indep snr

        snr_success = (snr >= snr_range[0]) * (snr <= snr_range[1])
        snr_attempt += 1
        if snr_success:
            pl_params["pl_snr"] = snr
            pl_params["pl_transits"] = 1+int((time[-1]-pl_params["pl_t0"])/pl_params["pl_period"])
        print("snr:", snr, f"({int(snr_success)*'accepted'}{(-int(snr_success)+1)*'rejected'})") if info else 0
    return transits, pl_params, snr_success


def get_lightcurve(num_planets, min_transits=2, period_range=(2,100), t_max=27.4, t_step=utils.min2day(2), 
                   time=None, pl_params=None, st_params=None, max_attempts=2, max_snr_attempts=5,
                   snr_range=(3,80), rdepth_range=(.25,5.), info=False, dur_range=(0,utils.hour2day(13)),
                   lower_snr=True, sigma=None):
    
    time = np.arange(0, t_max, t_step) if time is None else time
    st_given = None if st_params is None else st_params.copy()
    pl_given = None if pl_params is None else pl_params.copy()
    
    variability, st_params = get_stellar_variability(time, params=st_params)
    
    attempt, success = 0, False
    while not success and attempt < max_attempts:
        print("attempt:", attempt) if info else 0
        
        if st_given is None and attempt > 0:
            variability, st_params = get_stellar_variability(time, params=None)
        noise, sigma = get_photon_noise(time, sigma)
        background = variability + noise  # could be used for transit snr selection
        flux = background.copy()
        
        pls_params = {}
        pl_masks = np.zeros((num_planets, len(time)), dtype=bool)

        for i in range(num_planets):
            print("adding planet:", i+1) if info else 0
            
            pl_result = _add_planet(time, rdepth_range, snr_range, np.any(pl_masks, axis=0), 
                                    max_snr_attempts, pl_given, pl_params, sigma, min_transits, period_range, 
                                    i, info, dur_range, lower_snr)
            transits, pl_params, snr_success = pl_result
                                                                               
            if not snr_success:
                break  # stop adding planets
            # else... found the right planet to inject       
            pl_masks[i] = transits < 1
            pls_params[i] = pl_params.copy()
            flux *= transits
            
        if num_planets > 0 and not snr_success:
            attempt += 1
            continue  # try again with new background noise
        # else... found all the right planets, or no planets needed
        success = True
    if not success:
        return None, None, None, None
    for i in range(1, num_planets): # delete redundancies
        del pls_params[i]["st_u"], pls_params[i]["st_M"], pls_params[i]["st_R"]
    params = {**st_params, "sigma":sigma, "planets":pls_params, }
    return time, flux, pl_masks, params