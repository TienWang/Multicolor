

import numpy as np
import pandas as pd
import time
import os
import matplotlib.pyplot as plt
from vespa.transitsignal import TransitSignal_ASCII
from vespa.populations import HEBPopulation, EBPopulation
from vespa.populations import BEBPopulation, PlanetPopulation
from vespa.populations import PopulationSet
from vespa.stars.utils import fluxfrac, addmags
from vespa.transit_basic import MAInterpolationFunction
from isochrones.observation import ObservationTree, Observation, Source
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




def classification(df):
    planet_class = {'Earth':(0.8,1.25),
                'Super_Earth': (1.25,2),
                'Small_Neptunes': (2,4),
                'Large_Neptunes': (4,6),
                'Giants': (6,22) }
    m = {}
    for planet in planet_class:
        #print(planet,': ',((df>planet_class[planet][0])&((df<planet_class[planet][1]))).sum())
        m[planet]= (df>planet_class[planet][0])&((df<planet_class[planet][1]))

    return m

def plot_plpop_multiband_difference(stars):
    plt.figure(figsize=[15,25])
    percentiles = np.array([25, 50, 75])
    mask = classification(stars.obs_planet_radius)
    for inx, band in enumerate(['g','r','i','z','J','K','H']):
        df = abs(stars[band+'_depth']-stars.Kepler_depth)*1e6
        
        plt.subplot(4,2,inx+1)
        for m in mask:
            if mask[m].sum() != 0:
                n, bins,_ = plt.hist(df[mask[m]],histtype='step',lw=3,bins=50,label=m+':'+str(round(np.median(df[mask[m]]),1)))
                qua,med,three_quar =  np.percentile(df[mask[m]], percentiles)
                data['plpop_'+band+'_'+m+'_qua'] = qua
                data['plpop_'+band+'_'+m+'_med'] = med
                data['plpop_'+band+'_'+m+'_tqua'] = three_quar
            else:
                data['plpop_'+band+'_'+m+'_qua'] = 0
                data['plpop_'+band+'_'+m+'_med'] = 0
                data['plpop_'+band+'_'+m+'_tqua'] = 0
    
        plt.title(band+'-Kepler',size =20 )
        plt.xlabel('Depth Difference/ppm')
        plt.axvline(np.median(df),lw=3,c='r',alpha=0.5,ls = '--' ,label='Overall:' +str(round(np.median(df),0)))
        plt.legend(loc='center right')
        data['plpop_'+band+'-kep_med'] = round(np.median(df),1)
    plt.savefig(folder+'/plpop_multiband.jpg')
    plt.close()



    
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


def Calculate_multiband_transit(data,MAfn):

    #choose populations to calculate
    BEB = False
    TP = True
    HEB = False


    hostname = data['host_name']
    RA = data['RA']
    DEC = data['DEC']
    dist = data['distance']
    dist_err = data['distance_err']
    
    
    stellar_radius = data['stellar_radius']
    stellar_radius_err = data['stellar_radius_err']
    stellar_mass = data['stellar_mass']
    stellar_mass_err = data['stellar_mass_err']
    Teff = data['Teff']
    Teff_err = data['Teff_err']
    logg = data['logg']
    logg_err = data['logg_err']
    feh = data['feh']
    feh_err = data['feh_err']
    age = data['age']
    age_err = data['age_err']
    
    g_mag = data['g_mag']
    r_mag = data['r_mag']
    i_mag = data['i_mag']
    z_mag = data['z_mag']
    J_mag = data['J_mag']
    H_mag = data['H_mag']
    K_mag = data['K_mag']
    Kepler_mag = data['Kepler_mag']
    
    
    period = data['period']
    
    k,i = hostname.split('-')
    label = k+i
    
    '''
    period_err = data['period_err']
    planet_radius = data['planet_radius']
    '''

    mags = dict({'J': J_mag, 'g': g_mag , 'i': i_mag,
                     'K': K_mag, 'r': r_mag,
                     'Kepler': Kepler_mag , 'H': H_mag, 'z': z_mag})
    
    path = '/Users/neptune/Documents/Thesis/pipeline/data_4_28/'
    folder = path + label
    
    if not os.path.exists(folder):
        os.makedirs(folder)

    
    # Transiting Planet
    if TP:
        
        k,i = hostname.split('-')
        label = k+i
        mass = (stellar_mass, stellar_mass_err)
        radius = (stellar_radius, stellar_radius_err)
        distance = dist#pc
        parallex = (1000./distance,dist_err*1000./(distance ** 2))
        
        from isochrones.starmodel import StarModel, BinaryStarModel
        from isochrones import get_ichrone
        ichrone = get_ichrone('mist',bands = ['g','r','i','z','J','H','K','Kepler'])
        
        mags= dict({'J': (J_mag,0.05), 'g': (g_mag,0.05),
                    'i': (i_mag,0.05), 'z': (z_mag,0.05),
                    'K': (K_mag,0.05), 'r': (r_mag,0.05),
                    'Kepler': (Kepler_mag,0.05) ,'H': (H_mag,0.05)})
                        
        spec = { 'Teff': (Teff,Teff_err), 'logg':(logg,logg_err),
                 'feh':(feh,feh_err) }
        
        obs = ObservationTree(name = 'binary')
        for b,m in mags.items():
            o = Observation('',b,99)
            s = Source(m[0],m[1])
            o.add_source(s)
            obs.add_observation(o)
        
        obs.define_models(ic=ichrone,N=2)
        obs.add_spectroscopy(**spec)
        obs.add_parallax(parallex)
        starmodel = BinaryStarModel(ichrone,obs=obs)
        starmodel.fit(n_live_points=1500,refit=False,basename=label)
        starmodel.save_hdf(filename=folder+'/starmodel.h5',overwrite=True)
        
        data['plpop_teff_0'] = starmodel.random_samples(1e5).Teff_0_0.mean()
        data['plpop_teff_1'] = starmodel.random_samples(1e5).Teff_0_1.mean()
        data['plpop_mass_0'] = starmodel.random_samples(1e5).mass_0_0.mean()
        data['plpop_mass_1'] = starmodel.random_samples(1e5).mass_0_1.mean()
        data['plpop_radius_0'] = starmodel.random_samples(1e5).radius_0_0.mean()
        data['plpop_radius_1'] = starmodel.random_samples(1e5).radius_0_1.mean()
        data['plpop_Kepler_mag_0'] = starmodel.random_samples(1e5).Kepler_mag_0_0.mean()
        data['plpop_Kepler_mag_1'] = starmodel.random_samples(1e5).Kepler_mag_0_1.mean()
        data['plpop_Kepler_mag'] = addmags(starmodel.random_samples(1e5).Kepler_mag_0_0,
                                          +starmodel.random_samples(1e5).Kepler_mag_0_1).mean()
        data['plpop_K_mag_0'] = starmodel.random_samples(1e5).K_mag_0_0.mean()
        data['plpop_K_mag_1'] = starmodel.random_samples(1e5).K_mag_0_1.mean()
        data['plpop_K_mag'] = addmags(starmodel.random_samples(1e5).K_mag_0_0,
                                     +starmodel.random_samples(1e5).K_mag_0_1).mean()
        data['plpop_g_mag_0'] = starmodel.random_samples(1e5).g_mag_0_0.mean()
        data['plpop_g_mag_1'] = starmodel.random_samples(1e5).g_mag_0_1.mean()
        data['plpop_g_mag'] = addmags(starmodel.random_samples(1e5).g_mag_0_0,
                                     +starmodel.random_samples(1e5).g_mag_0_1).mean()
        data['plpop_z_mag_0'] = starmodel.random_samples(1e5).z_mag_0_0.mean()
        data['plpop_z_mag_1'] = starmodel.random_samples(1e5).z_mag_0_1.mean()
        data['plpop_z_mag'] = addmags(starmodel.random_samples(1e5).z_mag_0_0,
                                     +starmodel.random_samples(1e5).z_mag_0_1).mean()
        
        MAfn = MAInterpolationFunction(pmin=0.007, pmax=1, nzs=200, nps=400)
        plpop = PlanetPopulation(period = period, radius=radius,Teff = Teff,n=2e4,starmodel=starmodel,koi_dist = True,MAfn=MAfn)
        plpop.fit_trapezoids(MAfn=MAfn)
        
        
        '''
        if os.path.exists(folder+'/plpop_'+hostname+'.h5'):
            plpop = PlanetPopulation().load_hdf(filename = 'plpop_'+hostname+'.h5', path = folder)
        else:
            plpop = PlanetPopulation(period=period, rprs=rprs, mass=mass, radius=radius,n=2e4,starmodel=starmodel,MAfn=MAfn)
            plpop.fit_trapezoids(MAfn=MAfn)
            plpop.save_hdf(filename = 'plpop_'+hostname+'.h5', path = folder, overwrite=True)
        '''
    
        plpop.stars.drop(plpop.stars[plpop.stars.b_pri > 1].index, inplace=True)
        plpop.stars.reset_index(inplace=True,drop=True)
        plpop.stars['tru_planet_radius'] = plpop.stars.radius_2 *Rsun/Rear
        multi_depth = plpop.depth_in_band(['g','r','i','z','J','K','H','Kepler'])
        multi_depth['obs_planet_radius'] = np.sqrt(multi_depth.Kepler_depth)*plpop.stars.radius_1*Rsun/Rear
        for column in multi_depth.columns:
            plpop.stars[column]=multi_depth[column]
        
        plpop.stars.to_csv(folder + '/plpop_sample.csv',index=False)
        
        plt.figure(figsize=[15,25])
        percentiles = np.array([25, 50, 75])
        mask = classification(plpop.stars.obs_planet_radius)
        for inx, band in enumerate(['g','r','i','z','J','K','H']):
            df = abs(plpop.stars[band+'_depth']-plpop.stars.Kepler_depth)*1e6
            
            plt.subplot(4,2,inx+1)
            for m in mask:
                if mask[m].sum() != 0:
                    n, bins,_ = plt.hist(df[mask[m]],histtype='step',lw=3,bins=50,label=m+':'+str(round(np.median(df[mask[m]]),1)))
                    qua,med,three_quar =  np.percentile(df[mask[m]], percentiles)
                    data['plpop_'+band+'_'+m+'_qua'] = qua
                    data['plpop_'+band+'_'+m+'_med'] = med
                    data['plpop_'+band+'_'+m+'_tqua'] = three_quar
                else:
                    data['plpop_'+band+'_'+m+'_qua'] = 0
                    data['plpop_'+band+'_'+m+'_med'] = 0
                    data['plpop_'+band+'_'+m+'_tqua'] = 0
        
            plt.title(band+'-Kepler',size =20 )
            plt.xlabel('Depth Difference/ppm')
            plt.axvline(np.median(df),lw=3,c='r',alpha=0.5,ls = '--' ,label='Overall:' +str(round(np.median(df),0)))
            plt.legend(loc='center right')
            data['plpop_'+band+'-kep_med'] = round(np.median(df),1)
            
            norm_df = df/(plpop.stars.tru_planet_radius)**2
            qua,med,three_quar =  np.percentile(norm_df, percentiles)
            data['plpop_'+band+'-kep_qua_normalized'] = qua
            data['plpop_'+band+'-kep_med_normalized'] = med
            data['plpop_'+band+'-kep_tqua_normalized'] = three_quar
            
        plt.savefig(folder+'/plpop_multiband.jpg')
        plt.close()
        
        
        for inx,band in enumerate(['g','r','i','z','J','K','H']):
            
            dmag_0 = plpop.stars[band+'_mag_0'] - plpop.stars[band+'_mag_1']
            dmag_kep = plpop.stars['Kepler_mag_0'] - plpop.stars['Kepler_mag_1']
            data['plpop_'+band+'_dmag_med'] = round(np.median(abs(dmag_0-dmag_kep)),4)
        
        
    return data

    

'''
    # Hierarchical Eclipsing Binaries
    if HEB:


        if os.path.exists(folder+'/hebpop.h5'):
            hebpop = HEBPopulation().load_hdf(folder+'/hebpop.h5')
        else:
            from isochrones import get_ichrone
            from isochrones.starmodel import TripleStarModel
            mass = (stellar_mass, stellar_mass_err)
            radius = (stellar_radius, stellar_radius_err)
            period = period
            distance = dist#pc
            ichrone = get_ichrone('dartmouth',bands = ['g','r','i','z','J','H','K','Kepler'])
            params = dict({
                           'J': (J_mag,J_mag_err), 'g': (g_mag,0.05) , 'i': (i_mag,0.05),
                           'K': (K_mag,K_mag_err), 'r': (r_mag,0.05),
                           'Kepler': (Kepler_mag,0.05) , 'H': (H_mag,H_mag_err), 'z': (z_mag,0.05),
                           'parallex': (1000./distance,dist_err*1000./(distance ** 2))
                            })
            Tripstarmodel = TripleStarModel(ichrone,**params)
            Tripstarmodel.fit()
            
            hebpop = HEBPopulation(mass=mass, age=params['age'], feh=params['feh'],
                                   period=period,logg=params['logg'] ,mags=mags,starmodel=Tripstarmodel,n=2e4)
            hebpop.save_hdf(folder+'/hebpop.h5')
            
    
        if os.path.exists(folder + '/heb_band_depth.h5'):
            multi_color_depth = pd.read_hdf(folder + '/heb_band_depth.h5','table')
        else:
            multi_color_depth = depth_in_band_HEB(hebpop,['g','r','i','z','J','K','H','Kepler'])
            multi_color_depth.to_hdf(folder + '/heb_band_depth.h5', 'table', mode='w')
        
        
        multi_depth = multi_color_depth
        plt.figure(figsize=[15,25])
        for inx, band in enumerate(['g','r','i','z','J','K','H']):
            df = abs(multi_depth[band+'_depth']-multi_depth.Kepler_depth)*1e6 #ppm as unit
            df.dropna(inplace=True)
            plt.subplot(4,2,inx+1)
            n, bins,_ = plt.hist(df,histtype='step',lw=3,bins=50)
            plt.title(band+'-Kepler',size =20 )
            plt.xlabel('Depth Difference/ppm')
            plt.annotate('median value:'+str(round(np.median(df),1)),xy=(np.median(df),max(n)))
            plt.axvline(np.median(df))
            data['hebpop_'+band+'-kep_med'] = round(np.median(df),1)
        plt.savefig(folder+'/hebpop_multiband.jpg')
        plt.close()
    
        dmag_med = dict({})
        plt.figure(figsize=[15,25])
        for inx,band in enumerate(hebpop.mags):
            plt.subplot(4,2,inx+1)
            dmag = hebpop.dmag(band)
            dmag_med[band]=np.median(dmag)
            n, bins,_ = plt.hist(dmag,bins=50,histtype='step')
            plt.annotate('median value:'+str(round(np.median(dmag),3)),xy=(np.median(dmag),max(n)))
            plt.axvline(np.median(dmag))
            plt.title(band)
        
        plt.savefig(folder+'/hebpop_dmag.jpg')
        plt.close()
'''


