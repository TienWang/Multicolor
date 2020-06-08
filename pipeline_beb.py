

import numpy as np
import pandas as pd
import time
import os
import matplotlib.pyplot as plt
from vespa.populations import BEBPopulation, PlanetPopulation
from vespa.populations import PopulationSet
from vespa.transit_basic import MAInterpolationFunction
import logging
rootLogger = logging.getLogger()
rootLogger.setLevel(logging.INFO)
np.seterr(divide = 'ignore', invalid='ignore')
import warnings
warnings.simplefilter("error")
warnings.simplefilter("ignore", DeprecationWarning)
from astropy import constants as const
Rear = const.R_earth
Rsun = const.R_sun


def mag_range(mag,d1=1e-1,d2=1e-3):
    return mag-2.5*np.log10(d1),mag-2.5*np.log10(d2)



def depth_in_band_BEB(population,bands,MAfn):
    
    bandpass = [band+'_depth' for band in bands]
    multi_depth = pd.DataFrame(np.zeros([len(population.stars),len(bands)]),columns=bandpass)
    
    for band in bands:
        population.band = band
        
        
        for n in range(len(population.stars)):
            mag1 = population.stars.loc[n,'{}_mag_A'.format(band)]
            mag2 = population.stars.loc[n,'{}_mag_B'.format(band)]
            F1 = 10**(-0.4*mag1)
            F2 = 10**(-0.4*mag2)
            Ftot = F1+F2
            population.stars.loc[n,'fluxfrac_1'] = F1/Ftot
            population.stars.loc[n,'fluxfrac_2'] = F2/Ftot
        population.fit_trapezoids(MAfn=MAfn)
        dilution_factor = population.dilution_factor
        depth = population.stars['depth']
        multi_depth['{}_depth'.format(band)] = depth*dilution_factor
                 
    return multi_depth


    
def depth_in_band_HEB(population,bands,MAfn):

    bandpass = [band+'_depth' for band in bands]
    multi_depth = pd.DataFrame(np.zeros([len(population.stars),len(bands)]),columns=bandpass)
    
    for band in bands:
        population.band = band
        
        
        for n in range(len(population.stars)):
            mag2 = population.stars.loc[n,'{}_mag_B'.format(band)]
            mag3 = population.stars.loc[n,'{}_mag_C'.format(band)]
            F2 = 10**(-0.4*mag2)
            F3 = 10**(-0.4*mag3)
            Ftot = F2+F3
            population.stars.loc[n,'fluxfrac_1'] = F2/Ftot
            population.stars.loc[n,'fluxfrac_2'] = F3/Ftot
            
        population.fit_trapezoids(MAfn=MAfn)
        dilution_factor = population.dilution_factor
        depth = population.stars['depth']
        multi_depth['{}_depth'.format(band)] = depth*dilution_factor
                 
    return multi_depth

from astropy import units as u
from astropy.coordinates import SkyCoord
def radec(ra,dec):    
    return SkyCoord(ra,dec).ra.degree, SkyCoord(ra,dec).dec.degree


def Calculate_multiband_transit(data):

    #choose populations to calculate
    BEB = True

    hostname = data['koi']
    ra = data['RA']
    dec = data['DEC']
    g_mag = data['g_mag']
    r_mag = data['r_mag']
    i_mag = data['i_mag']
    z_mag = data['z_mag']
    J_mag = data['J_mag']
    H_mag = data['H_mag']
    K_mag = data['K_mag']
    Kepler_mag = data['Kepler_mag']
    period = data['period']
    
    mags = dict({'J': J_mag, 'g': g_mag , 'i': i_mag,
                     'K': K_mag, 'r': r_mag,
                     'Kepler': Kepler_mag , 'H': H_mag, 'z': z_mag})
    
    path = '/Users/neptune/Documents/Thesis/pipeline/data_beb_2/'
    folder = path + hostname
    
    if not os.path.exists(folder):
        os.makedirs(folder)

    if BEB:

        mags = dict({'J': J_mag, 'g': g_mag , 'i': i_mag,
                     'K': K_mag, 'r': r_mag,
                     'Kepler': Kepler_mag , 'H': H_mag, 'z': z_mag})
        
        n = 25000
        max_mag,min_mag = mag_range(mags['Kepler'])
        param = {'maglim':min_mag}
        MAfn = MAInterpolationFunction(pmin=0.007, pmax=2, nzs=400, nps=400)
        '''
        if os.path.exists(folder+'/beb_'+hostname+'.h5'):
            bebpop = BEBPopulation().load_hdf(folder+'/beb_'+hostname+'.h5')
        else:
            bebpop = BEBPopulation(trilegal_filename=folder + '/' + hostname + '_starfield.h5',
                               ra=ra, dec=dec, period=period, mags=mags,
                               n=n, **param)
            bebpop.stars = bebpop.stars[bebpop.stars.Kepler_mag > max_mag]
            bebpop.stars.reset_index(drop=True,inplace=True)
            bebpop.stars = bebpop.stars[bebpop.stars.Kepler_mag < min_mag]
            bebpop.stars.reset_index(drop=True,inplace=True)
            bebpop.stars.to_csv(folder + '/sample.csv',index=False)
            bebpop.save_hdf(folder+'/beb_'+hostname+'.h5', overwrite=True)
        
        '''
        bebpop = BEBPopulation(trilegal_filename=folder + '/' + hostname + '_starfield.h5',
                           ra=ra, dec=dec, period=period, mags=mags,
                           n=n, MAfn = MAfn, **param)
        bebpop.stars = bebpop.stars[bebpop.stars.Kepler_mag > max_mag]
        bebpop.stars.reset_index(drop=True,inplace=True)
        bebpop.stars = bebpop.stars[bebpop.stars.Kepler_mag < min_mag]
        bebpop.stars.reset_index(drop=True,inplace=True)
        bebpop.stars =bebpop.stars[bebpop.stars.b_pri < 2]
        bebpop.stars.reset_index(drop=True,inplace=True)
        
        bebpop.fit_trapezoids(MAfn=MAfn)
        bebpop.stars.to_csv(folder + '/sample.csv',index=False)
        bebpop.save_hdf(folder+'/beb_'+hostname+'.h5', overwrite=True)
        
        data['Teff_A'] = bebpop.stars.Teff_A.mean()
        data['Teff_B'] = bebpop.stars.Teff_B.mean()
        
        
        
        plt.figure()
        plt.hist(bebpop.stars.Kepler_mag_tot,histtype='step',lw=3,bins=25)
        plt.axvline(bebpop.mags['Kepler'],color='r',lw=3)
        plt.annotate('Target star Kepler mag: '+str(bebpop.mags['Kepler']),xy=(bebpop.mags['Kepler'],500))
        plt.title('Background Eclipsing Binaries total Kepler Magnitude Distribution',fontsize = 10)
        plt.savefig(folder+'/' + str(round(Kepler_mag,1)) + '_beb_kep_distribution.jpg')
        plt.close()
        
        data['beb_kepler_med'] = np.median(bebpop.stars.Kepler_mag_tot)

        plt.figure(figsize=[15,25])
        for inx,band in enumerate(bebpop.mags):
            plt.subplot(4,2,inx+1)
            
            dmag = bebpop.dmag(band)

            plt.hist(dmag,bins=50,histtype='step')
            plt.axvline(np.median(dmag))
            data['beb_' + band + '_dmag_med'] = np.median(dmag)
            plt.title(band)
        
        plt.savefig(folder+'/beb_dmag.jpg')
        plt.close()
        
        '''
        if os.path.exists(folder + '/beb_band_depth.h5'):
            multi_color_depth = pd.read_hdf(folder + '/beb_band_depth.h5','table')
        else:
            multi_color_depth = depth_in_band_BEB(bebpop,['g','r','i','z','J','K','H','Kepler'],MAfn)
            multi_color_depth.to_hdf(folder + '/beb_band_depth.h5', 'table', mode='w')
        '''
        multi_color_depth = depth_in_band_BEB(bebpop,['g','r','i','z','J','K','H','Kepler'],MAfn)
        multi_color_depth.to_hdf(folder + '/beb_band_depth.h5', 'table', mode='w')
        
        multi_depth = multi_color_depth
        plt.figure(figsize=[15,25])
        percentiles = np.array([40, 50, 60])
        for inx, band in enumerate(['g','r','i','z','J','K','H']):
        
            df = abs(multi_depth[band+'_depth']-multi_depth.Kepler_depth)*1e6
            df.dropna(inplace=True)
            plt.subplot(4,2,inx+1)
            n, bins,_ = plt.hist(df,histtype='step',lw=3,bins=50,range=[0,2000])
            plt.title(band+'-Kepler',size =20 )
            plt.xlabel('Depth Difference/ppm')
            plt.annotate('median value:'+str(round(np.median(df),1)),xy=(np.median(df),max(n)))
            plt.axvline(np.median(df))
            qua,med,three_quar =  np.percentile(df, percentiles)
            data['beb_'+band+'_qua'] = qua
            data['beb_'+band+'_med'] = med
            data['beb_'+band+'_tqua'] = three_quar
        
        plt.savefig(folder+'/beb_multiband.jpg')
        plt.close()
        
    return data

    

