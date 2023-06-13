import os, glob, rasterio, numpy as np, pandas as pd
from IPython.display import clear_output
    
def _calc_rayleigh_optical_depth(wavelength):

    '''
    This function calculates the Rayleigh optical depth for a given wavelength.

    Parameters: 
        wavelength (float): wavelength in micrometers.

    Returns:
        rayleigh_optical_depth (float): Rayleigh optical depth.     
    '''
    
    rayleigh_optical_depth = 0.008569 * (1 / (wavelength ** 4)) * (1 + 0.0113 * (1 / (wavelength ** 2)) 
                                                                   + 0.00013 * (1 / (wavelength ** 4)))
    return rayleigh_optical_depth


def _calc_rayleigh_reflectance(wavelength,
                               solar_azimuth_angle,
                               solar_zenith_angle,
                               sensor_azimuth_angle,
                               sensor_zenith_angle,
                               rayleigh_optical_depth):
    
    '''
    This function calculates the Rayleigh reflectance for a given wavelength.

    Parameters:
        wavelength (float): wavelength in micrometers.
        solar_azimuth_angle (float): solar azimuth angle in radians.
        solar_zenith_angle (float): solar zenith angle in radians.
        sensor_azimuth_angle (float): sensor azimuth angle in radians.
        sensor_zenith_angle (float): sensor zenith angle in radians.
        rayleigh_optical_depth (float): Rayleigh optical depth. Inherited from _calc_rayleigh_optical_depth() function.

    Returns:
        rayleigh_reflectance (float): Rayleigh reflectance.    
    '''
    
    rayleigh_phase = _calc_rayleigh_phase(solar_azimuth_angle = solar_azimuth_angle,
                                          solar_zenith_angle = solar_zenith_angle,
                                          sensor_azimuth_angle = sensor_azimuth_angle,
                                          sensor_zenith_angle = sensor_zenith_angle)
    
    us = np.cos(solar_zenith_angle)
    uv = np.cos(sensor_zenith_angle)
    air_mass = 1 / us + 1 / uv
    rayleigh_reflectance = (rayleigh_phase * (1 - np.exp(-1 * air_mass * rayleigh_optical_depth))) / (4 * (us + uv))
    return rayleigh_reflectance


def _calc_rayleigh_phase(solar_azimuth_angle,
                         solar_zenith_angle,
                         sensor_azimuth_angle,
                         sensor_zenith_angle):
    
    '''
    This function calculates the Rayleigh phase function for a given wavelength.

    Parameters:
        solar_azimuth_angle (float): solar azimuth angle in radians.
        solar_zenith_angle (float): solar zenith angle in radians.
        sensor_azimuth_angle (float): sensor azimuth angle in radians.
        sensor_zenith_angle (float): sensor zenith angle in radians.

    Returns:
        rayleigh_phase (float): Rayleigh phase function.    
    '''

    coef_a = 0.9587256
    coef_b = 1 - coef_a
    scattering_angle = _calc_scattering_angle(solar_azimuth_angle = solar_azimuth_angle,
                                              solar_zenith_angle=solar_zenith_angle,
                                              sensor_azimuth_angle=sensor_azimuth_angle,
                                              sensor_zenith_angle=sensor_zenith_angle)
    
    #rayleigh_phase = 3 * coef_a * (1 + np.cos(scattering_angle) ** 2) / (4 + coef_b)
    rayleigh_phase = ((3 * coef_a * (1 + np.cos(scattering_angle) ** 2)) / 4) + coef_b
    return rayleigh_phase


def _calc_scattering_angle(solar_azimuth_angle,
                           solar_zenith_angle,
                           sensor_azimuth_angle,
                           sensor_zenith_angle):
    
    '''
    This function calculates the scattering angle for a given wavelength.

    Parameters:
        solar_azimuth_angle (float): solar azimuth angle in radians.
        solar_zenith_angle (float): solar zenith angle in radians.
        sensor_azimuth_angle (float): sensor azimuth angle in radians.
        sensor_zenith_angle (float): sensor zenith angle in radians.

    Returns:
        scattering_angle (float): scattering angle.     
    '''

    relative_azimuth_angle = _calc_relative_azimuth_angle(angle_1 = solar_azimuth_angle,
                                                          angle_2 = sensor_azimuth_angle)
    
    scattering_angle = np.arccos(-1*np.cos(sensor_zenith_angle) * np.cos(solar_zenith_angle)
                                 + np.sin(sensor_zenith_angle) * np.sin(solar_zenith_angle) * np.cos(relative_azimuth_angle))
    return scattering_angle


def _calc_relative_azimuth_angle(angle_1,
                                 angle_2):
    
    '''
    This function calculates the relative azimuth angle for a given wavelength. 

    Parameters:
        angle_1 (float): angle 1 in radians.
        angle_2 (float): angle 2 in radians.

    Returns:
        relative_azimuth_angle (float): relative azimuth angle.    
    '''
    
    delta_phi = angle_1 - angle_2
    delta_phi = np.where(delta_phi > 2 * np.pi, delta_phi - 2 * np.pi, delta_phi)
    delta_phi = np.where(delta_phi < 0, delta_phi + 2 * np.pi, delta_phi)
    relative_azimuth_angle = np.abs(delta_phi - np.pi)
    return relative_azimuth_angle


def _calc_satm(rayleigh_optical_depth):

    '''
    This function calculates total atmospheric backscattering ratio.

    Parameters:
        rayleigh_optical_depth (float): Rayleigh optical depth. Inherited from _calc_rayleigh_optical_depth() function.

    Returns:
        satm (float): total atmospheric backscattering ratio.    
    '''
    
    satm = 0.92 * rayleigh_optical_depth * (np.exp(-1 * rayleigh_optical_depth))
    return satm


def _calc_total_transmittance(solar_zenith_angle,
                              sensor_zenith_angle,
                              rayleigh_optical_depth):
    
    '''
    This function calculates total transmittance. 
    Total transmittance is the product of transmittance on the surface-sensor path and transmittance on the surface-solar path.

    Parameters:
        solar_zenith_angle (float): solar zenith angle in radians.
        sensor_zenith_angle (float): sensor zenith angle in radians.
        rayleigh_optical_depth (float): Rayleigh optical depth. Inherited from _calc_rayleigh_optical_depth() function.

    Returns:
        total_transmittance (float): total transmittance.
    '''
    
    us = np.cos(solar_zenith_angle)
    uv = np.cos(sensor_zenith_angle)
    
    ts = np.exp(-1 * rayleigh_optical_depth / us) + np.exp(-1 * rayleigh_optical_depth / us) * (np.exp(0.52 * rayleigh_optical_depth / us) - 1)
    
    #print('\tCalculating Transmission on surface-sensor path ...')
    tv = np.exp(-1 * rayleigh_optical_depth / uv) + np.exp(-1 * rayleigh_optical_depth / uv) * (np.exp(0.52 * rayleigh_optical_depth / uv) - 1)
    
    total_transmittance = ts * tv
    return total_transmittance

def export_tif(opf, variable, var_name, band, param):

    '''
    This function exports the calculated variable to a GeoTIFF file.

    Parameters:
        opf (str): output folder path.
        variable (float): calculated variable.
        var_name (str): variable name.
        band (int): band number.
        param (dict): rasterio parameters.

    Returns:
        None.
    '''
    
    with rasterio.open(os.path.join(opf, f'{var_name}_{band}.TIF'), 'w', **param) as r:
        r.write(variable, 1)

def calc_quant(wavelength,
               solar_azimuth_angle_deg,
               solar_zenith_angle_deg,
               sensor_azimuth_angle_deg,
               sensor_zenith_angle_deg):
    
    '''
    This function calculates the Rayleigh optical depth, Rayleigh reflectance, and total atmospheric backscattering ratio for a given wavelength. 

    Parameters:
        wavelength (float): wavelength in micrometers.
        solar_azimuth_angle_deg (float): solar azimuth angle in degrees.
        solar_zenith_angle_deg (float): solar zenith angle in degrees.
        sensor_azimuth_angle_deg (float): sensor azimuth angle in degrees.
        sensor_zenith_angle_deg (float): sensor zenith angle in degrees.

    Returns:
        tau (float): Rayleigh optical depth.
        ray_ref (float): Rayleigh reflectance.
        satm (float): total atmospheric backscattering ratio.
        total_trans (float): total transmittance.
    '''
    
    solar_azimuth_angle = np.deg2rad(solar_azimuth_angle_deg)
    solar_zenith_angle = np.deg2rad(solar_zenith_angle_deg)
    sensor_azimuth_angle = np.deg2rad(sensor_azimuth_angle_deg)
    sensor_zenith_angle = np.deg2rad(sensor_zenith_angle_deg)
   
    tau = _calc_rayleigh_optical_depth(wavelength = wavelength)
   
    ray_ref = _calc_rayleigh_reflectance(wavelength = wavelength,
                                                      solar_azimuth_angle = solar_azimuth_angle,
                                                      solar_zenith_angle = solar_zenith_angle,
                                                      sensor_azimuth_angle = sensor_azimuth_angle,
                                                      sensor_zenith_angle = sensor_zenith_angle,
                                                      rayleigh_optical_depth = tau)
   
    satm = _calc_satm(rayleigh_optical_depth = tau)
   
    total_trans = _calc_total_transmittance(solar_zenith_angle = solar_zenith_angle,
                                                    sensor_zenith_angle = sensor_zenith_angle,
                                                    rayleigh_optical_depth = tau)
   
    return (tau, ray_ref, satm, total_trans)

def check_exist(inpf, band, sensor):

    '''
    This function checks if the input files exist.

    Parameters:
        inpf (str): input folder path.
        band (int): band number.
        sensor (str): sensor name. 

    Returns:
        response (bool): True if all the input files exist, False otherwise.
    '''
    
    response = False
    
    if sensor == 'L8':
        
        saa_band = glob.glob(os.path.join(inpf, '*_SAA.TIF'))[0]
        sza_band = glob.glob(os.path.join(inpf, '*_SZA.TIF'))[0]
        vaa_band = glob.glob(os.path.join(inpf, '*_VAA.TIF'))[0]
        vza_band = glob.glob(os.path.join(inpf, '*_VZA.TIF'))[0]
        
        files = [saa_band, sza_band, vaa_band, vza_band]
        exist = [f for f in files if os.path.isfile(f)]
        not_exist = list(set(exist) ^ set(files))
        
        if len(not_exist) == 0:
            response = True
        else:
            print(f"Files {not_exist} don't exist.")
            
    else:
        
        saa_band = os.path.join(inpf, 'sun_azimuth.img')
        sza_band = os.path.join(inpf, 'sun_zenith.img')
        vaa_band = os.path.join(inpf, f'view_azimuth_{band}.img')
        vza_band = os.path.join(inpf, f'view_zenith_{band}.img')
        
        files = [saa_band, sza_band, vaa_band, vza_band]
        exist = [f for f in files if os.path.isfile(f)]
        not_exist = list(set(exist) ^ set(files))
        
        if len(not_exist) == 0:
            response = True
        else:
            print(f"Files {not_exist} don't exist.")
        
    return response

def read_files(inpf, sensor, band):

    '''
    This function reads the input files. 

    Parameters:
        inpf (str): input folder path.
        sensor (str): sensor name.
        band (int): band number.

    Returns:
        saa (numpy.ndarray): solar azimuth angle.
        sza (numpy.ndarray): solar zenith angle.
        vaa (numpy.ndarray): sensor azimuth angle.
        vza (numpy.ndarray): sensor zenith angle.
        param (dict): rasterio parameters.
        shape (tuple): array shape.    
    '''
    
    if sensor == 'L8':
    
        saa_band = glob.glob(os.path.join(inpf, '*_SAA.TIF'))[0]
        sza_band = glob.glob(os.path.join(inpf, '*_SZA.TIF'))[0]
        vaa_band = glob.glob(os.path.join(inpf, '*_VAA.TIF'))[0]
        vza_band = glob.glob(os.path.join(inpf, '*_VZA.TIF'))[0]

        with (rasterio.open)(saa_band) as (r):
            saa = r.read(1).astype('float32')*0.01
            param = r.profile
        param.update(dtype = 'float32')
        shape = saa.shape
        saa[saa == 0] = np.nan

        with (rasterio.open)(sza_band) as (r):
            sza = r.read(1).astype('float32')*0.01
        sza[sza == 0] = np.nan

        with (rasterio.open)(vaa_band) as (r):
            vaa = r.read(1).astype('float32')*0.01
        vaa[vaa == 0] = np.nan

        with (rasterio.open)(vza_band) as (r):
            vza = r.read(1).astype('float32')*0.01
        vza[vza == 0] = np.nan
        
    else:

        saa_band = os.path.join(inpf, 'sun_azimuth.img')
        sza_band = os.path.join(inpf, 'sun_zenith.img')
        vaa_band = os.path.join(inpf, f'view_azimuth_{band}.img')
        vza_band = os.path.join(inpf, f'view_zenith_{band}.img')

        with (rasterio.open)(saa_band) as (r):
            saa = r.read(1).astype('float32')
            param = r.profile
        param.update(dtype = 'float32')
        shape = saa.shape
        saa[saa == 0] = np.nan

        with (rasterio.open)(sza_band) as (r):
            sza = r.read(1).astype('float32')
        sza[sza == 0] = np.nan

        with (rasterio.open)(vaa_band) as (r):
            vaa = r.read(1).astype('float32')
        vaa[vaa == 0] = np.nan

        with (rasterio.open)(vza_band) as (r):
            vza = r.read(1).astype('float32')
        vza[vza == 0] = np.nan


    return (saa, sza, vaa, vza, param, shape)
   
def rsr_calc(inpf, band, rsr_list, central_wavelength, export, sensor):

    '''
    This is the main function. It calculates the Rayleigh reflectance, transmittance, optical depth and atmospheric backscattering ratio at central wavelength as well 
    considering the relative spectral response (RSR) of the sensor. It also writes the outputs to a text file. 

    Parameters:
        inpf (str): input folder path.
        band (int): band number.
        rsr_list (list): a dataframe containing the relative spectral response (RSR) of the sensor.
        central_wavelength (int): central wavelength of the band.
        export (bool): if True, the outputs will be exported as GeoTIFF files.
        sensor (str): sensor name.

    Returns:
        rsr_rayleigh (numpy.ndarray): Rayleigh reflectance considering the RSR of the sensor.
        rsr_satm (numpy.ndarray): atmospheric backscattering ratio considering the RSR of the sensor.
        rsr_total_trans (numpy.ndarray): total transmittance considering the RSR of the sensor.
    
    '''
    
    opf = os.path.join(inpf, 'Output')
    os.makedirs(opf, exist_ok = True)
    
    saa, sza, vaa, vza, param, shape = read_files(inpf, sensor, band)
    
    print(f'\tProcessing at central wavelength: {central_wavelength*1000} nm...')
    tau_central, rayleigh_central, satm_central, trans_central = calc_quant(central_wavelength, saa, sza, vaa, vza)
    
    if export == True:
        export_tif(opf, rayleigh_central, 'rayleigh_reflectance', band, param)
        export_tif(opf, trans_central, 'transmittance', band, param)
    
    with open(os.path.join(opf, f'summary_central_wavelength_{band}.txt'), 'w') as r:
        r.write("central_wavelength: %d\nsatm: %f\ntau: %f\nmean_rayleigh_reflectance: %f\nmean_transmittance: %f\n" % 
                (central_wavelength*1000, satm_central, tau_central, np.nanmean(rayleigh_central), np.nanmean(trans_central)))
        r.write("min_rayleigh_reflectance: %f\nmax_rayleigh_reflectance: %f\nmin_transmittance: %f\nmax_transmittance: %f\n" % 
                (np.nanmin(rayleigh_central), np.nanmax(rayleigh_central), np.nanmin(trans_central), np.nanmax(trans_central)))
        
    print('\tDone.')
    print('\n\tProcessing relative spectral responses...')
    
    num_ray_ref = np.empty(shape)
    rays = []
    transs = []
    satms = []
    rsrs = []
    taus = []
    num_satm = float()
    num_total_trans = np.empty(shape)
    wavelengths = []
    deno = float()
    tau_sum = float()
    m = 1
   
    for wavelength, rsr in zip(rsr_list[band], rsr_list[f'rsr_{band}']):
        if pd.notnull(wavelength):
            per = (m/rsr_list[band].count())*100
            print(f"\tProcessing [{wavelength}nm]: [%-100s] %d%%" % ('='*int(1*per), 1*per), end = '\r', flush = True)
            m = m+1
            sim_tau, sim_ray_ref, sim_satm, sim_total_trans = calc_quant(wavelength*0.001, saa, sza, vaa, vza)
            num_ray_ref = num_ray_ref + sim_ray_ref*rsr
            rays.append(np.nanmean(sim_ray_ref))
            num_satm = num_satm + sim_satm*rsr
            wavelengths.append(wavelength)
            satms.append(sim_satm)
            rsrs.append(rsr)
            taus.append(sim_tau)
            tau_sum = tau_sum + sim_tau*rsr
            num_total_trans = num_total_trans + sim_total_trans*rsr
            transs.append(np.nanmean(sim_total_trans))
            deno = deno + rsr
        
    rsr_rayleigh = num_ray_ref/deno
    rsr_total_trans = num_total_trans/deno
    rsr_satm = num_satm/deno
    rsr_tau = tau_sum/deno
    
    with open(os.path.join(opf, f'rsr_summary_{band}.txt'), 'w') as r:
        r.write('wavelength: rsr: tau: simulated_satm: simulated_rayleigh: simulated_transmittance\n')
        for wavelength, rsr, tau, satm, ray, trans in zip(wavelengths, rsrs, taus, satms, rays, transs):
            r.write(f"%d: %f: %f: %f: %f: %f\n" % (wavelength, rsr, tau, satm, ray, trans))
        r.write('\nrsr_satm: %f\nrsr_tau: %f\nrsr_mean_rayleigh: %f\nrsr_mean_transmittance: %f\n' % 
                (rsr_satm, rsr_tau, np.nanmean(rsr_rayleigh), np.nanmean(rsr_total_trans)))
        r.write('\nrsr_min_rayleigh: %f\nrsr_max_rayleigh: %f\nrsr_min_transmittance: %f\nrsr_max_transmittance: %f' % 
                (np.nanmin(rsr_rayleigh), np.nanmax(rsr_rayleigh), np.nanmin(rsr_total_trans), np.nanmax(rsr_total_trans)))
        r.write('\ndifference[satm - rsr_satm]: %f\ndifference[tau - rsr_tau]: %f\ndifference[central_rayleigh - rsr_rayeligh]: %f\ndifference[central_transmittance - rsr_transmittance]: %f\n' % 
                ((satm_central - rsr_satm), (tau_central - rsr_tau), (np.nanmean(rayleigh_central) - np.nanmean(rsr_rayleigh)), 
                 (np.nanmean(trans_central) - np.nanmean(rsr_total_trans))))
    
    if export == True:
        export_tif(opf, rsr_rayleigh, 'rsr_rayleigh_reflectance', band, param)
        export_tif(opf, rsr_total_trans, 'rsr_transmittance', band, param)
        export_tif(opf, (rayleigh_central - rsr_rayleigh), 'diff_rayleigh', band, param)
        export_tif(opf, (trans_central - rsr_total_trans), 'diff_transmittance', band, param)
    
    print(f'\n{band}: Done.')
    return (rsr_rayleigh, rsr_satm, rsr_total_trans)

def do_rsr(export):

    '''
    This is the caller function for the RSR calculation function. It takes the sensor name, band choice, and folder path containing the 
    radiance and metadata (if any) file as input. It takes the input as prompts. 

    Parameters: 
        export (bool): If True, the RSR reflectance and transmittance will be exported as GeoTIFF files. Default is False.

    Returns:
        None
    '''
    
    if export is None:
        export = False
    
    sensor = input('Enter the sensor name [Landsat 8 (1), Sentinel 2A (2) or Sentinel 2B (3)]: ')
    band_choice = input('To calculate for all bands, enter "All". For specific bands, enter band number [e.g. B1]: ')
    if (sensor == 'Landsat 8' or sensor == '1'):
        inpf = input('Enter the folder containing Landsat 8 radiance and MTL.txt file:\n')
        rsr_list = pd.read_csv("https://raw.githubusercontent.com/akhi9661/just_some_data/main/rsr_l8.txt",
                               sep = ",",
                               on_bad_lines = "skip")
        
        central_wavelength = {'B1': 0.443, 'B2': 0.482, 'B3': 0.5615, 'B4': 0.6545, 'B5': 0.865, 'B6': 1.6085} #'B7': 2.2005
        
        if ((band_choice != 'All') and (band_choice != 'all')):
            if check_exist(inpf, band_choice, sensor = 'L8'):
                print(f'\nProcessing [{band_choice}]...')
                rsr_rayleigh, rsr_satm, rsr_total_trans = rsr_calc(inpf, band_choice, rsr_list, central_wavelength[band_choice], export, sensor = 'L8')
                clear_output(wait = True)
            else:
                do_rsr()  
        else:
            n = 1
            for band in ['B1', 'B2', 'B3', 'B4', 'B5', 'B6']: #'B7'
                if check_exist(inpf, band, sensor = 'L8'):
                    print(f'\nProcessing [{band}]: {n}/7...')
                    rsr_rayleigh, rsr_satm, rsr_total_trans = rsr_calc(inpf, band, rsr_list, central_wavelength[band], export, sensor = 'L8')
                    n = n + 1
                    clear_output(wait = True)
                else:
                    n = n + 1
                    continue
            
    elif (sensor == 'Sentinel 2A' or sensor == '2'):
        inpf = input('Enter the folder containing radiance and angle files:\n')
        rsr_list = pd.read_csv("https://raw.githubusercontent.com/akhi9661/just_some_data/main/rsr_s2a.txt",
                               sep = ",",
                               on_bad_lines = "skip")
        
        central_wavelength = {'B1': 0.4427, 'B2': 0.4924, 'B3': 0.5598, 'B4': 0.6646, 'B5': 0.7041, 'B6': 0.7405, 'B7': 0.7828,
                              'B8': 0.8328, 'B8A': 0.8647, 'B9': 0.9451, 'B10': 1.3735, 'B11': 1.6137, 'B12': 2.2024}
        
        if ((band_choice != 'All') and (band_choice != 'all')):
            
            if check_exist(inpf, band_choice, sensor = 'S2'):
                print(f'\nProcessing [{band_choice}]...')
                rsr_rayleigh, rsr_satm, rsr_total_trans = rsr_calc(inpf, band_choice, rsr_list, central_wavelength[band_choice], export, sensor = 'S2')
                clear_output(wait = True)     
            else: 
                do_rsr()

        else:
            n = 1
            for band in ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9', 'B10', 'B11', 'B12']:
                if check_exist(inpf, band, sensor = 'S2'):
                    print(f'\nProcessing [{band}]: {n}/13...')
                    rsr_rayleigh, rsr_satm, rsr_total_trans = rsr_calc(inpf, band, rsr_list, central_wavelength[band], export, sensor = 'S2')
                    n = n + 1
                    clear_output(wait = True)
                else:
                    n = n + 1
                    continue
    
    elif (sensor == 'Sentinel 2B' or sensor == '3'):
        inpf = input('Enter the folder containing radiance and angle files:\n')
        rsr_list = pd.read_csv("https://raw.githubusercontent.com/akhi9661/just_some_data/main/rsr_s2b.txt",
                               sep = ",",
                               on_bad_lines = "skip")
        
        central_wavelength = {'B1': 0.4422, 'B2': 0.4921, 'B3': 0.5590, 'B4': 0.6649, 'B5': 0.7038, 'B6': 0.7391, 'B7': 0.7797,
                              'B8': 0.8329, 'B8A': 0.8640, 'B9': 0.9432, 'B10': 1.3769, 'B11': 1.6104, 'B12': 2.1857}
        
        if ((band_choice != 'All') and (band_choice != 'all')):
            if check_exist(inpf, band_choice, sensor = 'S2'):
                print(f'\nProcessing [{band_choice}]...')
                rsr_rayleigh, rsr_satm, rsr_total_trans = rsr_calc(inpf, band_choice, rsr_list, central_wavelength[band_choice], export, sensor = 'S2')
                clear_output(wait = True)     
            else: 
                do_rsr()

        else:
            n = 1
            for band in ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9', 'B10', 'B11', 'B12']:
                if check_exist(inpf, band, sensor = 'S2'):
                    print(f'\nProcessing [{band}]: {n}/13...')
                    rsr_rayleigh, rsr_satm, rsr_total_trans = rsr_calc(inpf, band, rsr_list, central_wavelength[band], export, sensor = 'S2')
                    n = n + 1
                    clear_output(wait = True)
                else:
                    n = n + 1
                    continue
   
    else: 
        print('Invalid sensor name. Try again.')
        do_rsr()
        
    return None

# Example ------
do_rsr(input('Export outputs as TIF files [may take a lot of space. Default is False]: '))
