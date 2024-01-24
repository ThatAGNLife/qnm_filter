import matplotlib.pyplot as plt
import numpy as np
import astropy.constants as c
import qnm_filter
import seaborn as sns
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from matplotlib.ticker import MultipleLocator
import sxs
import scipy
from scipy.interpolate import interp1d
import sys
import lal

def harmonics_to_polarizations_complex(data_complex, iota, beta, model_list) -> None: #for complex data class (i.e. filtered waveforms)

    strain = 0
    for l, m in model_list:
        ylm = lal.SpinWeightedSphericalHarmonic(iota, beta, -2, l, m)
        strain += data_complex[str(l) + str(m)] * ylm
    time = data_complex[str(l) + str(m)].time
    ifo = data_complex[str(l) + str(m)].ifo
    hp = np.real(strain)
    hc = -np.imag(strain)
    return {
        "plus": qnm_filter.RealData(hp, index=time, ifo=ifo),
        "cross": qnm_filter.RealData(hc, index=time, ifo=ifo),
    }

def get_NR_strain(bilby_ifo, parameters, NRwaveform):
    """Construct a NR strain
    
    Parameters
    ----------
    bilby_ifo : bilby.gw.detector.Interferometer
        An instance of `bilby.gw.detector.Interferometer`.
    iota : float
        inclination angle in the source frame, in radian.
    beta : float
        azimuth angle in the source frame, in radian.
    ra : float
        right ascension, in radian.
    dec : float
        declination, in radian.
    psi : float
        polarization angle, in radian.
        
    Returns
    -------
    qnm_filter.RealData
        Strain data
    """
    iota = parameters['iota']
    beta = parameters['beta']
    ra = parameters['ra']
    dec = parameters['dec']
    psi = parameters['psi']
    nr_dataset = NRwaveform.harmonics_to_polarizations("data_in_si", iota, beta, [(2,2), (2,-2), (3,3), (3,-3)])
    hplus = nr_dataset['plus']
    hcross = nr_dataset['cross']
    hplus_interp_func = interp1d(hplus.time, hplus.values, bounds_error=False, fill_value=0)
    hcross_interp_func = interp1d(hcross.time, hcross.values, bounds_error=False, fill_value=0)

    time = bilby_ifo.strain_data.time_array
    hplus = hplus_interp_func(time)
    hcross = hcross_interp_func(time)
    
    fp = bilby_ifo.antenna_response(ra, dec, time[0], psi, 'plus')
    fc = bilby_ifo.antenna_response(ra, dec, time[0], psi, 'cross')

    return qnm_filter.RealData(hplus*fp+hcross*fc, index=time, ifo=bilby_ifo.name)

def error_profile(error_param, param, strain, lines = [], plots = True): 
    name = error_param['name']
    centre = error_param['centre']
    width = error_param['width']
    amp = error_param['amp']
    angle = error_param['phase']
    
    bbh_m = param['bbh_m']
    rem_m = param['rem_m']
    freq_list = param['freq_list']
    
    if name == 'normal_a_p':
        sig = width/4
        PDF = scipy.stats.norm.pdf(strain.fft_freq, centre, sig)
        RRMag = PDF/max(PDF)*(amp-1) + 1
        RRPhase = np.zeros(len(strain.fft_freq)) + PDF/max(PDF)*np.deg2rad(angle)
        error = RRMag *np.exp(1j*RRPhase)
        print(name + ' loaded')
        lb = centre-70
        ub = centre+70
    else: 
        print('profile not found')
    
    if plots == True:
        fig, (ax1, ax2) = plt.subplots(1,2)
        fig.suptitle(f'Error Profile: Frequency Centre ($\omega_M$) - {centre:.4f} Hz; Width - {width} Hz')
        colours = plt.cm.tab20(np.linspace(0, 1, len(lines)+2))
        ax1.plot(strain.fft_freq, np.abs(error), color = colours[0])
        ax2.plot(strain.fft_freq, np.angle(error,deg=True), color = colours[0])
        if len(lines) > 0:
            w = np.linspace(lb,ub, 1000)
            for i in range(0,len(lines)):
                F = abs(1/(w-freq_list[lines[i]]))
                ax1.plot(w, F/max(F)*(amp-1) + 1, color = colours[i+1])
                ax2.plot(w, F/max(F)*angle, color = colours[i+1])
                ax1.axvline(freq_list[lines[i]].real,label=lines[i], color = colours[i+1])
                ax2.axvline(freq_list[lines[i]].real,label=lines[i], color = colours[i+1])
        ax1.set_xlim(lb,ub)
        ax1.set_ylabel('Amplitude')
        ax2.set_ylabel(r'Angle ($\theta$)')
        ax2.set_xlim(lb,ub)
        ax1.set_title('Amplitude')
        ax2.set_title('Phase')
        ax1.legend()
        fig.supxlabel(r'Frequency (Hz)')
        fig.tight_layout()
    else:
         pass
    return error

def error_profile_data(data_loaded_freq, data_loaded_mag, data_loaded_phase, param, strain, lines = [], plots = True): 
    
    #bbh_m = param['bbh_m']
    #rem_m = param['rem_m']
    freq_list = param['freq_list']
    
    fmag = interp1d(data_loaded_freq, data_loaded_mag, bounds_error=False, fill_value=1)
    fphase = interp1d(data_loaded_freq, data_loaded_phase, bounds_error=False, fill_value=0)
    mag = fmag(strain.fft_freq)
    phase = fphase(strain.fft_freq)
    error = mag *np.exp(1j*phase)
    lb = 5
    ub = 5000
    
    if plots == True:
        fig, (ax1, ax2) = plt.subplots(1,2)
        fig.suptitle(f'Error Profile')
        colours = plt.cm.tab20(np.linspace(0, 1, len(lines)+2))
        ax1.plot(strain.fft_freq, np.abs(error), color = colours[0])
        ax2.plot(strain.fft_freq, np.angle(error,deg=True), color = colours[0])
        if len(lines) > 0:
            w = np.linspace(lb,ub, 1000)
            for i in range(0,len(lines)):
                F = abs(1/(w-freq_list[lines[i]]))
                ax1.plot(w, F/max(F)*(max(np.abs(error))-1) + 1, color = colours[i+1])
                ax2.plot(w, F/max(F)*max(np.angle(error,deg=True)), color = colours[i+1])
                ax1.axvline(freq_list[lines[i]].real,label=lines[i], color = colours[i+1])
                ax2.axvline(freq_list[lines[i]].real,label=lines[i], color = colours[i+1])
        ax1.set_xlim(lb,ub)
        ax1.set_ylabel('Amplitude')
        ax2.set_ylabel(r'Angle ($\theta$)')
        ax2.set_xlim(lb,ub)
        ax1.set_title('Amplitude')
        ax2.set_title('Phase')
        ax1.legend()
        ax1.set_xscale('log')
        ax1.set_yscale('log')
        ax2.set_xscale('log')
        ax2.set_yscale('log')
        fig.supxlabel(r'Frequency (Hz)')
        fig.tight_layout()
    else:
         pass
    return error


def bias_strain(strain, error):
    #strain is already padded
    #fft strain and bias strain by multiplying the error
    biased_strain_fft_data = error*strain.fft_data 
    #now we fft back into the time domain
    biased_strain_data = np.fft.irfft(biased_strain_fft_data, norm="ortho", n=len(strain.time))
    #create a Real Data instance with this data
    biased_strain = qnm_filter.RealData(biased_strain_data, index=strain.time)
    return biased_strain

def filter_exact(parameters, strain, modes):
    fit = qnm_filter.Network(segment_length=parameters['segment_length'], srate=parameters['srate'], t_init = parameters['t_init'])
    fit.original_data['H1'] = strain
    fit.add_filter(mass=parameters['rem_m']*parameters['bbh_m'], chi=parameters['chi'], model_list=modes)
    return fit.filtered_data['H1']

def find_likelihood(parameters, biased_strain, noise, strain, modes = [(2, 2, 0, "p"),(2, 2, 1, "p"),(3, 3, 0, "p")], mass_inc = 1, delta_inc = 0.01): #put mass param into function (for changing bbh_m)
    fit = qnm_filter.Network(segment_length=parameters['segment_length'], srate=parameters['srate'], t_init = parameters['t_init'])
    fit.original_data['H1'] = biased_strain
    fit.detector_alignment()
    fit.pure_noise = {}
    fit.pure_noise['H1'] = noise#*error
    fit.pure_nr = {}
    fit.pure_nr['H1'] = strain #for SNR purposes
    fit.condition_data('original_data')
    fit.condition_data('pure_noise')
    fit.condition_data('pure_nr')
    fit.compute_acfs('pure_noise')
    fit.acfs['H1']*=0
    fit.acfs['H1'][0]=parameters['sigma']**2
    fit.cholesky_decomposition()
    fit.first_index()
    delta_mass = mass_inc
    delta_chi = delta_inc
    massspace = np.arange(34, 100, delta_mass)
    chispace = np.arange(0.0, 0.95, delta_chi)
    mass_grid, chi_grid = np.meshgrid(massspace, chispace)
    #modes = [(2, 2, 0, "p"),(2, 2, 1, "p")]
    likelihood_data, _ = qnm_filter.parallel_compute(fit, massspace, chispace, 
                                                 num_cpu = -1 ,model_list = modes)
    credible_region = qnm_filter.find_credible_region(likelihood_data)
    return likelihood_data, credible_region, mass_grid, chi_grid

def find_SNR(parameters, biased_strain, noise, strain):
    fit = qnm_filter.Network(segment_length=parameters['segment_length'], srate=parameters['srate'], t_init = 0)
    fit.original_data['H1'] = biased_strain
    fit.detector_alignment()
    fit.pure_noise = {}
    fit.pure_noise['H1'] = noise
    fit.pure_nr = {}
    fit.pure_nr['H1'] = strain #for SNR purposes
    fit.condition_data('original_data')
    fit.condition_data('pure_noise')
    fit.condition_data('pure_nr')
    fit.compute_acfs('pure_noise')
    fit.acfs['H1']*=0
    fit.acfs['H1'][0]=parameters['sigma']**2
    fit.cholesky_decomposition()
    fit.first_index()
    SNR = fit.compute_SNR(fit.truncate_data(fit.original_data)['H1'], fit.truncate_data(fit.pure_nr)['H1'], 'H1', False)
    return SNR

def likelihood_pair(parameters, signal_noise, signal_no_noise, error, modes = [(2, 2, 0, "p"),(2, 2, 1, "p"),(3,3,0,'p')], mass_inc = 1, delta_inc = 0.01, title = 'Blank', filter_modes = []):
    biased_strain = bias_strain(signal_no_noise, error)
    if filter_modes:
        biased_strain_final = filter_exact(parameters, biased_strain, filter_modes)
        strain_final = filter_exact(parameters, signal_no_noise, filter_modes)
    else: 
        biased_strain_final = biased_strain
        strain_final = signal_no_noise
        
    likelihood_data_biased, credible_region_biased, mass_grid, chi_grid = find_likelihood(parameters, biased_strain_final, signal_noise, strain_final, modes)
    likelihood_data_unbiased, credible_region_unbiased, _, _ = find_likelihood(parameters, strain_final, signal_noise, strain_final, modes)
    likelihood_pair = {'biased': [likelihood_data_biased, credible_region_biased, biased_strain], 'unbiased': [likelihood_data_unbiased, credible_region_unbiased, signal_no_noise], 'grid': [mass_grid, chi_grid], 'title': title}
    return likelihood_pair