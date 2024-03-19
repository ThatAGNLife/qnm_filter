import matplotlib.pyplot as plt
import numpy as np
import astropy.constants as c
from .Network import *
from .gw_data import *
from .utility import *
from .bilby_helper import *
from .sxs_helper import *
import seaborn as sns
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from matplotlib.ticker import MultipleLocator
import sxs
import scipy
from scipy.interpolate import interp1d
import sys
import lal

def get_freq_list_si(freq_list, parameters):

    """Converts the QNM frequency peak dictionary from simulation to SI units.
    
    Parameters
    ----------
    freq_list : dict
        Dictionary containing QNM peak frequencies in SI units.
    param : dict
        Dictionary containing base parameters. This function uses rem_m and bbh_m.
            rem_m : float
                Remnant mass in terms of total binary system mass.
            bbm_h : float
                Total binary system mass in solar mass units.

    Returns
    -------
    dict
        Dictionary of QNM frequency peaks in SI units.
    """

    freq_list_si = {}
    keyslist = list(freq_list.keys())
    for i in keyslist:
        freq_sim = freq_list[i]
        freq_si = freq_sim/Filter.mass_unit(parameters['rem_m']*parameters['bbh_m'])/2/np.pi
        freq_list_si[i] = freq_si
    return freq_list_si

def get_NR_strain(bilby_ifo, parameters, NRwaveform, modes):

    """Construct a NR strain from a SXS waveform.
    
    Parameters
    ----------
    bilby_ifo : bilby.gw.detector.Interferometer
        An instance of `bilby.gw.detector.Interferometer`.
    parameters : dict
        Dictionary containing base parameters. This function uses iota, beta, ra, dec, and psi.
            iota : float
                Inclination angle in the source frame, in radians.
            beta : float
                Azimuth angle in the source frame, in radians.
            ra : float
                Right ascension, in radians.
            dec : float
                Declination, in radians.
            psi : float
                Polarization angle, in radians.
    NRwaveform : qnm_filter.sxs_helper.SXSWaveforms
        An instance of 'qnm_filter.sxs_helper.SXSWaveforms' - a pre-loaded SXS waveform.
    modes : list
        List of l, m modes in this waveform to unpack and include in the strain, e.g. [(2, 2), (2, -2), (3, 3)].
        
    Returns
    -------
    qnm_filter.RealData
        Strain conversion for the SXS file.
    """

    iota = parameters['iota']
    beta = parameters['beta']
    ra = parameters['ra']
    dec = parameters['dec']
    psi = parameters['psi']
    nr_dataset = NRwaveform.harmonics_to_polarizations("data_in_si", iota, beta, modes)
    hplus = nr_dataset['plus']
    hcross = nr_dataset['cross']
    hplus_interp_func = interp1d(hplus.time, hplus.values, bounds_error=False, fill_value=0)
    hcross_interp_func = interp1d(hcross.time, hcross.values, bounds_error=False, fill_value=0)

    time = bilby_ifo.strain_data.time_array
    hplus = hplus_interp_func(time)
    hcross = hcross_interp_func(time)
    
    fp = bilby_ifo.antenna_response(ra, dec, time[0], psi, 'plus')
    fc = bilby_ifo.antenna_response(ra, dec, time[0], psi, 'cross')

    return RealData(hplus*fp+hcross*fc, index=time, ifo=bilby_ifo.name)

def error_profile(error_param, param, strain, lines = [], plots = True): 

    """Construct an artifical Gaussian calibration error profile according to the input parameters.
    
    Parameters
    ----------
    error_param : dict
        Dictionary containing base parameters for the calibration error shape. This function uses name, centre, width, amp, and phase.
            name : string
                Type of error. Currently only houses one type: normal_a_p.
    param : dict
        Dictionary containing base parameters. This function uses freq_list.
            freq_list : dict
                Contains the frequency peak information for each QNM mode specified in Hz.
    strain : qnm_filter.RealData
        This function uses the strain.fft_freq in order to output the error values at the correct frequencies for multiplication.
    lines : list
        Contains names of the QNM as strings that are to be shown on the plot.
    plots : True/False
        Outputs plot if set to True.
        
    Returns
    -------
    numpy.ndarray
        1-D array containing the calibration error to be multiplied by the fft_data of a qnm_filter.RealData class. Same frequency units as the input strain qnm_filter.RealData class.
    """

    name = error_param['name']
    centre = error_param['centre']*2*np.pi
    width = error_param['width']*2*np.pi
    amp = error_param['amp']
    angle = error_param['phase']
    freq_list = param['freq_list']
    
    if name == 'normal_a_p':
        sig = width/4
        PDF = scipy.stats.norm.pdf(strain.fft_freq, centre, sig)
        RRMag = PDF/max(PDF)*(amp-1) + 1
        RRPhase = np.zeros(len(strain.fft_freq)) + PDF/max(PDF)*np.deg2rad(angle)
        error = RRMag *np.exp(1j*RRPhase)
        print(name + ' loaded')
        lb = centre/(2*np.pi)-70
        ub = centre/(2*np.pi)+70
    else: 
        print('profile not found')
    
    if plots == True:
        fig, (ax1, ax2) = plt.subplots(1,2)
        fig.suptitle(f'Error Profile: Frequency Centre ($\omega_M$) - {centre/(2*np.pi):.4f} Hz; Width - {width/(2*np.pi)} Hz', fontsize = 12)
        colours = plt.cm.tab20(np.linspace(0, 1, len(lines)+2))
        ax1.plot(strain.fft_freq/(2*np.pi), np.abs(error), color = colours[0])
        ax2.plot(strain.fft_freq/(2*np.pi), np.angle(error,deg=True), color = colours[0])
        if len(lines) > 0:
            w = np.linspace(lb,ub, 1000)
            for i in range(0,len(lines)):
                F = abs(1/(w-freq_list[lines[i]]))
                ax1.plot(w, F/max(F)*(amp-1) + 1, color = colours[i+1])
                ax2.plot(w, F/max(F)*angle, color = colours[i+1])
                ax1.axvline(freq_list[lines[i]].real,label=lines[i], color = colours[i+1])
                ax2.axvline(freq_list[lines[i]].real,label=lines[i], color = colours[i+1])
        ax1.set_xlim(lb,ub)
        ax1.set_ylabel('Amplitude', fontsize = 12)
        ax2.set_ylabel(r'Angle ($\theta$)', fontsize = 12)
        ax2.set_xlim(lb,ub)
        ax1.set_title('Amplitude', fontsize = 12)
        ax2.set_title('Phase', fontsize = 12)
        ax1.legend()
        fig.supxlabel(r'Frequency (Hz)', fontsize = 12)
        ax1.tick_params(axis='both', which='major', labelsize=12)
        ax2.tick_params(axis='both', which='major', labelsize=12)
        fig.tight_layout()
    else:
         pass
    return error

def error_profile_data(data_loaded_freq, data_loaded_mag, data_loaded_phase, param, strain, lines = [], plots = True): 

    """Construct a physical calibration error profile interpolated + extrapolated from the loaded data.
    
    Parameters
    ----------
    data_loaded_freq : list
        List of frequencies there are magnitude and phase calibration error values for.
    data_loaded_mag : list
        List of magnitudes for the calibration error. 
    data_loaded_phase : list
        List of phase for the calibration error.   
    param : dict
        Dictionary containing base parameters. This function uses freq_list.
            freq_list : dict
                Contains the frequency peak information for each QNM mode specified in Hz.
    strain : qnm_filter.RealData
        This function uses the strain.fft_freq in order to output the error values at the correct frequencies for multiplication.
    lines : list
        Contains names of the QNM as strings that are to be shown on the plot.
    plots : True/False
        Outputs plot if set to True.
        
    Returns
    -------
    numpy.ndarray
        1-D array containing the calibration error to be multiplied by the fft_data of a qnm_filter.RealData class. Same frequency units as the input strain qnm_filter.RealData class.
    """
    
    #bbh_m = param['bbh_m']
    #rem_m = param['rem_m']
    freq_list = param['freq_list']
    
    fmag = interp1d(data_loaded_freq, data_loaded_mag, bounds_error=False, fill_value=1)
    fphase = interp1d(data_loaded_freq, data_loaded_phase, bounds_error=False, fill_value=0)
    mag = fmag(strain.fft_freq/(2*np.pi))
    phase = fphase(strain.fft_freq/(2*np.pi))
    error = mag *np.exp(1j*phase)
    lb = 5
    ub = 5000
    
    if plots == True:
        fig, (ax1, ax2) = plt.subplots(1,2)
        fig.suptitle(f'Error Profile', fontsize = 12)
        colours = plt.cm.tab20(np.linspace(0, 1, len(lines)+2))
        ax1.plot(strain.fft_freq/(2*np.pi), np.abs(error), color = colours[0])
        ax2.plot(strain.fft_freq/(2*np.pi), np.angle(error,deg=True), color = colours[0])
        if len(lines) > 0:
            w = np.linspace(lb,ub, 1000)
            for i in range(0,len(lines)):
                F = abs(1/(w-freq_list[lines[i]]))
                ax1.plot(w, F/max(F)*(max(np.abs(error))-1) + 1, color = colours[i+1])
                ax2.plot(w, F/max(F)*max(np.angle(error,deg=True)), color = colours[i+1])
                ax1.axvline(freq_list[lines[i]].real,label=lines[i], color = colours[i+1])
                ax2.axvline(freq_list[lines[i]].real,label=lines[i], color = colours[i+1])
        ax1.set_xlim(lb,ub)
        ax1.set_ylabel('Amplitude', fontsize = 12)
        ax2.set_ylabel(r'Angle ($\theta$)', fontsize = 12)
        ax2.set_xlim(lb,ub)
        ax1.set_title('Amplitude', fontsize = 12)
        ax2.set_title('Phase', fontsize = 12)
        ax1.legend()
        ax1.set_xscale('log')
        ax1.set_yscale('log')
        ax2.set_xscale('log')
        ax2.set_yscale('log')
        fig.supxlabel(r'Frequency (Hz)', fontsize = 12)
        ax1.tick_params(axis='both', which='major', labelsize=12)
        ax2.tick_params(axis='both', which='major', labelsize=12)
        fig.tight_layout()
    else:
         pass
    return error

def bias_strain(strain, error):

    """Bias a waveform by the input error.
    
    Parameters
    ----------
    strain : qnm_filter.RealData
        Input waveform.
    error : numpy.ndarray
        1-D array containing the calibration error to be multiplied by the fft_data of a qnm_filter.RealData class. Ensure frequency values are the same by using the error creation function correctly.
        
    Returns
    -------
    qnm_filter.RealData
        Strain biased by input calibration error.
    """

    #strain is already padded
    #fft strain and bias strain by multiplying the error
    biased_strain_fft_data = error*strain.fft_data 
    #now we fft back into the time domain
    biased_strain_data = np.fft.irfft(biased_strain_fft_data, norm="ortho", n=len(strain.time))
    #create a Real Data instance with this data
    biased_strain = RealData(biased_strain_data, index=strain.time)
    return biased_strain

def filter_exact(parameters, strain, modes):

    """Exactly filter/remove QNM modes from the strain.
    
    Parameters
    ----------
    bilby_ifo : bilby.gw.detector.Interferometer
        An instance of `bilby.gw.detector.Interferometer`.
    parameters : dict
        Dictionary containing base parameters. This function uses ifo, rem_m, bbh_m, chi.
            ifo : string
                Interferometer used for the antenna pattern, fit function, etc. 
            rem_m : float
                Remnant mass in terms of total binary system mass.
            bbm_h : float
                Total binary system mass in solar mass units.
            chi : float
                Dimensionless spin of remnant.
    strain : qnm_filter.RealData
        Strain signal to be filtered.
    modes : list
        List of QNM modes to filter/remove, e.g. [(2, 2, 0, "p"), (2, 2, 1, "p"), (3 ,3 , 0,'p')].
        
    Returns
    -------
    qnm_filter.RealData
        Filtered strain.
    """

    fit = Network()
    fit.original_data[parameters['ifo']] = strain
    fit.add_filter(mass=parameters['rem_m']*parameters['bbh_m'], chi=parameters['chi'], model_list=modes)
    return fit.filtered_data[parameters['ifo']]

def find_likelihood_no_noise(parameters, biased_strain, noise, strain, modes, mass_inc = 1, delta_inc = 0.01, mass_bounds = [34, 100], chi_bounds = [0, 0.95], cores = -1): #put mass param into function (for changing bbh_m)
    
    """Find the likelihood and credible region for plotting and calculating the predicted mass and chi values.

    changed
    
    Parameters
    ----------
    parameters : dict
        Dictionary containing base parameters. This function uses segment_length, srate, t_init, sigma.
            segment_length : float
                Segment length of signal.
            srate : float
                Sampling rate.
            t_init : float
                Starting time of the segment of the signal to be investigated.
            sigma : float
                With no noise injections the noise acfs needs to be manually inputted. This will decide the SNR.
    biased_strain : qnm_filter.RealData
        Strain biased by the calibration error.
    noise : qnm_filter.RealData
        White noise for SNR purposes.
    strain : qnm_filter.RealData
        Unbiased signal strain.
    modes : list
        List of free modes to be fitted for.
    mass_inc : float
        Mass grid width.
    delta_inc : float
        Spin grid width.
    mass_bounds : list
        Two values listing the min. and max. values for the mass grid.
    chi_bounds : list
        Two values listing the min. and max. values for the chi grid.
        
    Returns
    -------
    numpy.ndarray
        Likelihood regions.
    numpy.ndarray
        90% credible region.
    numpy.ndarray
        Mass mesh grid for plotting.
    numpy.ndarray
        Chi mesh grid for plotting.
    """

    fit = Network(segment_length=parameters['segment_length'], srate=parameters['srate'], t_init = parameters['t_init'])
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
    massspace = np.arange(mass_bounds[0], mass_bounds[1], delta_mass)
    chispace = np.arange(chi_bounds[0], chi_bounds[1], delta_chi)
    mass_grid, chi_grid = np.meshgrid(massspace, chispace)
    likelihood_data, _ = parallel_compute(fit, massspace, chispace, 
                                                 num_cpu = cores ,model_list = modes)
    #print(likelihood_data[0])
    credible_region = find_credible_region(likelihood_data, num_cpu = cores)
    quan = posterior_quantile_2d(likelihood_data, fit, parameters['rem_m']*parameters['bbh_m'], parameters['chi'], model_list = modes, num_cpu = cores)
    return likelihood_data, credible_region, mass_grid, chi_grid, quan, fit

def find_likelihood(parameters, biased_strain, noise, strain, modes, mass_inc = 1, delta_inc = 0.01, mass_bounds = [34, 100], chi_bounds = [0, 0.95], cores = -1): #put mass param into function (for changing bbh_m)
    
    """Find the likelihood and credible region for plotting and calculating the predicted mass and chi values.

    changed
    
    Parameters
    ----------
    parameters : dict
        Dictionary containing base parameters. This function uses segment_length, srate, t_init, sigma.
            segment_length : float
                Segment length of signal.
            srate : float
                Sampling rate.
            t_init : float
                Starting time of the segment of the signal to be investigated.
            sigma : float
                With no noise injections the noise acfs needs to be manually inputted. This will decide the SNR.
    biased_strain : qnm_filter.RealData
        Strain biased by the calibration error.
    noise : qnm_filter.RealData
        White noise for SNR purposes.
    strain : qnm_filter.RealData
        Unbiased signal strain.
    modes : list
        List of free modes to be fitted for.
    mass_inc : float
        Mass grid width.
    delta_inc : float
        Spin grid width.
    mass_bounds : list
        Two values listing the min. and max. values for the mass grid.
    chi_bounds : list
        Two values listing the min. and max. values for the chi grid.
        
    Returns
    -------
    numpy.ndarray
        Likelihood regions.
    numpy.ndarray
        90% credible region.
    numpy.ndarray
        Mass mesh grid for plotting.
    numpy.ndarray
        Chi mesh grid for plotting.
    """

    fit = Network(segment_length=parameters['segment_length'], srate=parameters['srate'], t_init = parameters['t_init'])
    fit.original_data[parameters['ifo']] = biased_strain
    fit.detector_alignment()
    fit.pure_noise = {}
    fit.pure_noise[parameters['ifo']] = noise#*error: error already biased when input into function
    fit.pure_nr = {}
    fit.pure_nr[parameters['ifo']] = strain #for SNR purposes
    fit.condition_data('original_data')
    fit.condition_data('pure_noise')
    fit.condition_data('pure_nr')
    fit.compute_acfs('pure_noise')
    fit.cholesky_decomposition()
    fit.first_index()
    delta_mass = mass_inc
    delta_chi = delta_inc
    massspace = np.arange(mass_bounds[0], mass_bounds[1], delta_mass)
    chispace = np.arange(chi_bounds[0], chi_bounds[1], delta_chi)
    mass_grid, chi_grid = np.meshgrid(massspace, chispace)
    likelihood_data, _ = parallel_compute(fit, massspace, chispace, 
                                                 num_cpu = cores ,model_list = modes)
    #print(likelihood_data[0])
    credible_region = find_credible_region(likelihood_data, num_cpu = cores)
    quan = posterior_quantile_2d(likelihood_data, fit, parameters['rem_m']*parameters['bbh_m'], parameters['chi'], model_list = modes, num_cpu = cores)
    return likelihood_data, credible_region, mass_grid, chi_grid, quan, fit

def find_SNR_no_noise(parameters, biased_strain, noise, strain):

    """Calculates the total SNR from the beginning of the merger. 
    
    Parameters
    ----------
    parameters : dict
        Dictionary containing base parameters. This function uses segment_length, srate, t_init, sigma.
            segment_length : float
                Segment length of signal.
            srate : float
                Sampling rate.
    biased_strain : qnm_filter.RealData
        Strain biased by the calibration error.
    noise : qnm_filter.RealData
        White noise for SNR purposes.
    strain : qnm_filter.RealData
        Unbiased signal strain.

    Returns
    -------
    float
        Returns SNR.
    """
    
    fit = Network(segment_length=parameters['segment_length'], srate=parameters['srate'], t_init = 0)
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

def find_SNR(parameters, biased_strain, noise, strain):

    """Calculates the total SNR from the beginning of the merger. 
    
    Parameters
    ----------
    parameters : dict
        Dictionary containing base parameters. This function uses segment_length, srate, t_init, sigma.
            segment_length : float
                Segment length of signal.
            srate : float
                Sampling rate.
    biased_strain : qnm_filter.RealData
        Strain biased by the calibration error.
    noise : qnm_filter.RealData
        White noise for SNR purposes.
    strain : qnm_filter.RealData
        Unbiased signal strain.

    Returns
    -------
    float
        Returns SNR.
    """
    
    fit = Network(segment_length=parameters['segment_length'], srate=parameters['srate'], t_init = 0)
    fit.original_data[parameters['ifo']] = biased_strain
    fit.detector_alignment()
    fit.pure_noise = {}
    fit.pure_noise[parameters['ifo']] = noise
    fit.pure_nr = {}
    fit.pure_nr[parameters['ifo']] = strain #for SNR purposes
    fit.condition_data('original_data')
    fit.condition_data('pure_noise')
    fit.condition_data('pure_nr')
    fit.compute_acfs('pure_noise')
    fit.cholesky_decomposition()
    fit.first_index()
    SNR = fit.compute_SNR(fit.truncate_data(fit.original_data)[parameters['ifo']], fit.truncate_data(fit.pure_nr)[parameters['ifo']], parameters['ifo'], False)
    return SNR

def likelihood_pair_no_noise(parameters, signal_noise, signal_no_noise, error, modes, mass_inc = 1, delta_inc = 0.01, title = 'Blank', filter_modes = [], mass_bounds = [34, 100], chi_bounds = [0, 0.95], cores = -1):
    
    """Returns likelihood, and credible regions for the biased and unbiased strain pair.
    
    Parameters
    ----------
    parameters : dict
        Dictionary containing base parameters. This function uses segment_length, srate, t_init, sigma, rem_m, bbh_m, chi.
            rem_m : float
                Remnant mass in terms of total binary system mass.
            bbm_h : float
                Total binary system mass in solar mass units.
            chi : float
                Dimensionless spin of remnant.
            segment_length : float
                Segment length of signal.
            srate : float
                Sampling rate.
            t_init : float
                Starting time of the segment of the signal to be investigated.
            sigma : float
                With no noise injections the noise acfs needs to be manually inputted. This will decide the SNR.
    noise : qnm_filter.RealData
        White noise for SNR purposes.
    signal_no_noise : qnm_filter.RealData
        Unbiased signal strain.
    error : numpy.ndarray
        1-D array containing the calibration error to be multiplied by the fft_data of a qnm_filter.RealData class. Ensure frequency values are the same by using the error creation function correctly.
    modes : list
        List of free modes to be fitted for.
    mass_inc : float
        Mass grid width.
    delta_inc : float
        Spin grid width.
    title : string
        Adds a title the resulting dictionary for plotting purposes: e.g. Free: 220 + 221 modes
    filter_modes : list
        List of modes to be exactly filtered from the signal before calculating the likelihood. Modes in this list and the modes list cannot overlap.
    mass_bounds : list
        Two values listing the min. and max. values for the mass grid.
    chi_bounds : list
        Two values listing the min. and max. values for the chi grid.
            
    Returns
    -------
    dict
        Creates a likelihood pair dictionary that includes the biased signal's likelihood grid and 90% credible interval, the same for the unbiased signal, the mass and chi mesh grid for plotting and the title, also for plotting purposes.
    """

    biased_strain = bias_strain(signal_no_noise, error)

    if filter_modes:
        biased_strain_final = filter_exact(parameters, biased_strain, filter_modes)
        strain_final = filter_exact(parameters, signal_no_noise, filter_modes)
    else: 
        biased_strain_final = biased_strain
        strain_final = signal_no_noise
        
    likelihood_data_biased, credible_region_biased, mass_grid, chi_grid, quan_biased, fit_biased = find_likelihood_no_noise(parameters, biased_strain_final, signal_noise, strain_final, modes, mass_inc, delta_inc, mass_bounds, chi_bounds, cores = cores)
    likelihood_data_unbiased, credible_region_unbiased, _, _, guan_unbiased, fit_unbiased = find_likelihood_no_noise(parameters, strain_final, signal_noise, strain_final, modes, mass_inc, delta_inc, mass_bounds, chi_bounds, cores = cores)
    likelihood_pair = {'biased': [likelihood_data_biased, credible_region_biased, biased_strain_final, quan_biased, fit_biased], \
                       'unbiased': [likelihood_data_unbiased, credible_region_unbiased, strain_final, guan_unbiased, fit_unbiased], 'grid': [mass_grid, chi_grid], 'title': title}
    return likelihood_pair

def likelihood_pair(parameters, signal_noise, signal_with_noise, signal_without_noise, error, modes, mass_inc = 1, delta_inc = 0.01, title = 'Blank', filter_modes = [], mass_bounds = [34, 100], chi_bounds = [0, 0.95], cores = -1):
    
    """Returns likelihood, and credible regions for the biased and unbiased strain pair.
    
    Parameters
    ----------
    parameters : dict
        Dictionary containing base parameters. This function uses segment_length, srate, t_init, sigma, rem_m, bbh_m, chi.
            rem_m : float
                Remnant mass in terms of total binary system mass.
            bbm_h : float
                Total binary system mass in solar mass units.
            chi : float
                Dimensionless spin of remnant.
            segment_length : float
                Segment length of signal.
            srate : float
                Sampling rate.
            t_init : float
                Starting time of the segment of the signal to be investigated.
            sigma : float
                With no noise injections the noise acfs needs to be manually inputted. This will decide the SNR.
    noise : qnm_filter.RealData
        White noise for SNR purposes.
    signal_no_noise : qnm_filter.RealData
        Unbiased signal strain.
    error : numpy.ndarray
        1-D array containing the calibration error to be multiplied by the fft_data of a qnm_filter.RealData class. Ensure frequency values are the same by using the error creation function correctly.
    modes : list
        List of free modes to be fitted for.
    mass_inc : float
        Mass grid width.
    delta_inc : float
        Spin grid width.
    title : string
        Adds a title the resulting dictionary for plotting purposes: e.g. Free: 220 + 221 modes
    filter_modes : list
        List of modes to be exactly filtered from the signal before calculating the likelihood. Modes in this list and the modes list cannot overlap.
    mass_bounds : list
        Two values listing the min. and max. values for the mass grid.
    chi_bounds : list
        Two values listing the min. and max. values for the chi grid.
            
    Returns
    -------
    dict
        Creates a likelihood pair dictionary that includes the biased signal's likelihood grid and 90% credible interval, the same for the unbiased signal, the mass and chi mesh grid for plotting and the title, also for plotting purposes.
    """

    biased_strain = bias_strain(signal_with_noise, error)
    biased_noise = bias_strain(signal_noise, error)

    if filter_modes:
        biased_strain_final = filter_exact(parameters, biased_strain, filter_modes)
        strain_final = filter_exact(parameters, signal_with_noise, filter_modes)
    else: 
        biased_strain_final = biased_strain
        strain_final = signal_with_noise
        
    likelihood_data_biased, credible_region_biased, mass_grid, chi_grid, quan_biased, fit_biased = find_likelihood(parameters, biased_strain_final, biased_noise, signal_without_noise, modes, mass_inc, delta_inc, mass_bounds, chi_bounds, cores=cores)
    likelihood_data_unbiased, credible_region_unbiased, _, _, guan_unbiased, fit_unbiased = find_likelihood(parameters, strain_final, biased_noise, signal_without_noise, modes, mass_inc, delta_inc, mass_bounds, chi_bounds, cores=cores)
    likelihood_pair = {'biased': [likelihood_data_biased, credible_region_biased, biased_strain_final, quan_biased, fit_biased], 'unbiased': [likelihood_data_unbiased, credible_region_unbiased, strain_final, guan_unbiased, fit_unbiased], 'grid': [mass_grid, chi_grid], 'title': title}
    return likelihood_pair