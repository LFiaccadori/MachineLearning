import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import pickle
from astropy.io import fits
from astropy import coordinates as coord, units as u
from astropy.time import Time
import sys

if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")

if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")



class TemporaryAperturePhotometry:
    # Initialization method
    def __init__(self):
        """
        In questa funzione vengono definiti i parametri di base per inizializzare la classe.
        """
    
        self.data_path = './group10_WASP-135_20190803/'
        
        self.readout_noise = 7.4
        
        self.gain = 1.91
        self.bias_std = 1.3
        
        self.median_bias = pickle.load(open('./median_bias.p', 'rb'))
        self.median_bias_error = pickle.load(open('./median_bias_error.p', 'rb'))

        self.median_normalized_flat = pickle.load(open('./median_normalized_flat.p', 'rb'))
        self.median_normalized_flat_error = pickle.load(open('./median_normalized_flat_error.p', 'rb'))
        
        self.science_path = self.data_path + 'science/'
        self.science_list = np.genfromtxt('./group10_WASP-135_20190803/science/science.list', dtype=str)

        self.science_size = len(self.science_list)
        #print('La lunghezza della science list è', self.science_size)
        #group10_WASP-135_20190803/science/science_list.txt
        
        ylen, xlen = np.shape(self.median_bias)
        X_axis = np.arange(0, xlen, 1.)
        Y_axis = np.arange(0, ylen, 1.)
        self.X, self.Y = np.meshgrid(X_axis, Y_axis)

    def provide_aperture_parameters(self, sky_inner_radius, sky_outer_radius, aperture_radius, x_initial, y_initial):
        """
        Funzione per caricare i parametri necessari a effettuare aperture photometry.
        
        Parameters:
        - sky_inner_radius: raggio interno per l'annulus selection.
        - sky_outer_radius: raggio esterno per l'annulus selection.
        - aperture_radius: raggio di apertura per aperture photometry.
        - x_initial: coordinata x dell'oggetto.
        - y_initial: coordinata y dell'oggetto.
        """
        
        self.sky_inner_radius = sky_inner_radius 
        self.sky_outer_radius = sky_outer_radius 
        self.aperture_radius = aperture_radius 
        self.x_initial = x_initial
        self.y_initial = y_initial

    def compute_centroid(self, science_corrected, x_target_initial, y_target_initial, maximum_number_of_iterations=20):

        for i_iter in range(0, maximum_number_of_iterations):

            if i_iter == 0:
                # first iteration
                x_target_previous = x_target_initial
                y_target_previous = y_target_initial
            else:
                # using the previous result as starting point
                x_target_previous = x_target_refined
                y_target_previous = y_target_refined

            # 2D array with the distance of each pixel from the target star 
            target_distance = np.sqrt((self.X-x_target_previous)**2 + (self.Y-y_target_previous)**2)

            # Selection of the pixels within the inner radius
            annulus_sel = (target_distance < self.sky_inner_radius)
            
            # Weighted sum of coordinates
            weighted_X = np.sum(science_corrected[annulus_sel]*self.X[annulus_sel])
            weighted_Y = np.sum(science_corrected[annulus_sel]*self.Y[annulus_sel])

            # Sum of the weights
            total_flux = np.sum(science_corrected[annulus_sel])

            # Refined determination of coordinates
            x_target_refined = weighted_X/total_flux
            y_target_refined = weighted_Y/total_flux

            percent_variance_x = (x_target_refined-x_target_previous)/(x_target_previous) * 100.
            percent_variance_y = (y_target_refined-y_target_previous)/(y_target_previous) * 100.
            # exit condition: both percent variance are smaller than 0.1%
            if np.abs(percent_variance_x)<0.1 and  np.abs(percent_variance_y)<0.1:
                break

        return x_target_refined, y_target_refined

    def compute_sky_background(self, science_corrected, x_pos, y_pos):
        target_distance = np.sqrt((self.X-x_pos)**2 + (self.Y-y_pos)**2)

        annulus_selection = (target_distance > self.sky_inner_radius) & (target_distance<=self.sky_outer_radius)

        sky_flux_average = np.sum(science_corrected[annulus_selection]) / np.sum(annulus_selection)
        sky_flux_median = np.median(science_corrected[annulus_selection])

        N_sky = np.sum(annulus_selection)
        sky_std = np.std(science_corrected[annulus_selection])

        sky_flux_average_error = np.std(science_corrected[annulus_selection])/np.sqrt(N_sky)
        sky_flux_median_error = 1.2* sky_flux_average_error 

        return sky_flux_median, sky_flux_median_error, annulus_selection
   
    def julian_date(self):
        
        from astropy import coordinates as coord, units as u        
        target = coord.SkyCoord("17:49:08.37","+29:52:44", unit=(u.hourangle, u.deg), frame='icrs')

        from astropy.time import Time

        # Information regarding the light travel time calculation (ultima parte)
        j_d = self.julian_date + self.exptime/86400./2. 

        tm = Time(j_d, format='jd', scale='utc', location=('45.8472d', '11.569d')) 

        ltt_bary = tm.light_travel_time(target)  

        bjd_tdb = tm.tdb + ltt_bary

        return bjd_tdb
            
    def aperture_photometry(self):

        self.airmass = np.empty(self.science_size)
        self.exptime = np.empty(self.science_size)
        self.julian_date = np.empty(self.science_size)

        self.aperture = np.empty(self.science_size)
        self.aperture_errors = np.empty(self.science_size)
        self.sky_background = np.empty(self.science_size)
        self.sky_background_errors = np.empty(self.science_size)

        self.x_position = np.empty(self.science_size)
        self.y_position = np.empty(self.science_size)

        for i_science, science_name in enumerate(self.science_list):

            science_fits = fits.open(self.science_path + science_name)
            self.airmass[i_science] = science_fits[0].header['AIRMASS']
            self.exptime[i_science] = science_fits[0].header['EXPTIME']
            self.julian_date[i_science] = science_fits[0].header['JD']

            
            science_data = science_fits[0].data * self.gain
            science_corrected, science_corrected_error = self.correct_science_frame(science_data)
            # save the data from the first HDU 
            science_fits.close()
            
        
            x_refined, y_refined = self.compute_centroid(science_corrected, self.x_initial, self.y_initial)

            self.sky_background[i_science], self.sky_background_errors[i_science], annulus_selection = \
            self.compute_sky_background(science_corrected, x_refined, y_refined)

            science_sky_corrected = science_corrected - self.sky_background[i_science]
            science_sky_corrected_error = np.sqrt((science_corrected_error[annulus_selection])**2 + (self.sky_background_errors[i_science])**2)

            
            target_distance = np.sqrt((self.X-x_refined)**2 + (self.Y-y_refined)**2)
            aperture_selection = (target_distance < self.aperture_radius)
            

            

            self.aperture[i_science] =  np.sum(science_sky_corrected[aperture_selection])

            # calcolo l'errore associato alla aperture photometry
            aperture_flux_average = np.sum(science_corrected[aperture_selection]) / np.sum(aperture_selection)
            aperture_flux_median = np.median(science_corrected[aperture_selection])

            N_sky = np.sum(aperture_selection)
            
            aperture_std = np.std(science_corrected[aperture_selection])

            aperture_flux_average_error = np.std(science_corrected[aperture_selection])/np.sqrt(N_sky)
            aperture_flux_median_error = 1.2* aperture_flux_average_error

            

            self.x_position[i_science] = x_refined
            self.y_position[i_science] = y_refined      

    def correct_science_frame(self, science_data):
        science_debiased = science_data - self.median_bias
        science_corrected = science_debiased / self.median_normalized_flat

        # Error associated to the science corrected frame
        science_debiased_errors = np.sqrt(self.readout_noise**2 + self.median_bias_error**2 + science_debiased)
        science_corrected_errors = science_corrected * np.sqrt((science_debiased_errors/science_debiased)**2
                                                            + (self.median_normalized_flat_error/self.median_normalized_flat)**2)
        return science_corrected, science_corrected_errors