#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 13 09:56:37 2019
#goal : investigate impact of chirp table settings on radar sensitivity and higher moments
# input:
# - Sw noise threshold to use for filtering out radar noise (suggested values: 0.01 , 0.025, 0.038)
# - chirpMode: string identifying the chirp mode selected for plotting
# - site: location for the observations
# - PathRadarData: path where ncdf radar Wband files are located
# - pathFig: path where figures are going to be saved
# - flags for executing  (=1)(or not (=0)) a specific plot: 
    flagPlotScatterplots          = 0 -> scatter plots of higher moments against reflectivity
    flagplot_SKnoNoise            = 0 -> time height plot of skewness without noise
    flagplot_SK                   = 1 -> time height plot of skewness
    flagplot_SWnoNoise            = 1 -> time height plot of spectrum width without noise
    flagplot_SW                   = 1 -> time height plot of spectrum width
    flagplot_VDnoNoise            = 1 -> time height plot of mean Doppler velocity without noise
    flagplot_VD                   = 1 -> time height plot of mean Doppler velocity
    flagplot_ZEnoNoise            = 1 -> time height plot of reflectivity without noise
    flagplot_ZE                   = 1 -> time height plot of reflectivity
    flagPlot_histogramSensitivity = 1 -> plot of Ze distributions per height
    flagPlotDistribMomentsZE      = 1 -> plot of distributions of Ze values
    flagPlotDistribMomentsVD      = 1 -> plot of distributions of Vd values 
    flagPlotDistribMomentsSW      = 1 -> plot of distributions of Sw values
    flagPlotDistribMomentsSK      = 1 -> plot of distributions of Sk values
# @author    : Claudia Acquistapace
# institution: Institute for geophysics and meteorology - University of Cologne
# Address    : Pohligstrasse 3, 50969 Koeln
# email      : cacquist@meteo.uni-koeln.de
# phone      : +49 221 470 6276
"""



import numpy as np
import matplotlib
import scipy
import pylab
import netCDF4 as nc4
import numpy.ma as ma
import matplotlib.pyplot as plt
from netCDF4 import Dataset
import struct
import glob
import pandas as pd
import datetime as dt
import random
import datetime
import matplotlib.dates as mdates
from datetime import timedelta
from functionsWband import f_calcCloudBaseTop
from functionsWband import convertWbandTimeUnits

# todo : 
# 1) understand how to convert time in datetime format 
# 2) create an ordered dictionary in python with series of Ze, Vd, Sw, Sk values
# 3) 
 
# definition of users parameters
SWNoiseThr                    =  0.025#0.038#0.025#0.01 # threshold in spectral width for removing noise
chirpMode                     = 'PBLmode'#'HRmode'
SWThresholdString             = 'thr_025'
site                          = 'BCO'
day                           = '20190208'
outString                     = day+'_'+site+'_'+chirpMode+'_'+SWThresholdString
pathRadarData                 = '/work/cacquist/BARBADOS_WBAND_OBS/20190208/' #
#Users/cacquist/Lavoro/proposal_ship_DFG_barbados/Barbados_data/BARBADOS_WBAND_OBS/20190207/'
pathFig                       = '/work/cacquist/BARBADOS_WBAND_OBS/figs/new_names/'
fileOutTxt                    = 'fit_sensitivity_'+outString

# definitions of flags for plotting graphics
flagPlotScatterplots          = 0
flagplot_SKnoNoise            = 0
flagplot_SK                   = 1
flagplot_SWnoNoise            = 1
flagplot_SW                   = 1 
flagplot_VDnoNoise            = 1
flagplot_VD                   = 1
flagplot_ZEnoNoise            = 1
flagplot_ZE                   = 1
flagPlot_histogramSensitivity = 1
flagPlotDistribMomentsZE      = 1
flagPlotDistribMomentsVD      = 1
flagPlotDistribMomentsSW      = 1
flagPlotDistribMomentsSK      = 1




#fileList                      = glob.glob(pathRadarData+'*LV1.NC')


HRmode                        = ['190208_180000_P06_ZEN.LV1.NC', '190208_190000_P06_ZEN.LV1.NC',\
                                 '190208_200000_P06_ZEN.LV1.NC', '190208_210000_P06_ZEN.LV1.NC']
PBLmode                       = ['190208_000001_P09_ZEN.LV1.NC', '190208_010001_P09_ZEN.LV1.NC','190208_020001_P09_ZEN.LV1.NC',\
                                 '190208_030000_P09_ZEN.LV1.NC', '190208_040000_P09_ZEN.LV1.NC','190208_050000_P09_ZEN.LV1.NC',\
                                 '190208_060001_P09_ZEN.LV1.NC', '190208_070000_P09_ZEN.LV1.NC','190208_080000_P09_ZEN.LV1.NC',\
                                 '190208_090000_P09_ZEN.LV1.NC', '190208_100000_P09_ZEN.LV1.NC','190208_110001_P09_ZEN.LV1.NC',\
                                 '190208_120001_P09_ZEN.LV1.NC', '190208_130000_P09_ZEN.LV1.NC','190208_140001_P09_ZEN.LV1.NC']
#'190208_171325_P06_ZEN.LV1.NC', 
nFilesHR                      = len(HRmode)
nFilesPBL                     = len(PBLmode)
nFiles                        = nFilesPBL
dictionary_array              = []
timeAll                       = []
LWP_arr                       = []
CBH_arr                       = []

# loop on the files to read time arrays and series of columnar values and store them on a global file
for indFile in range(nFiles):
    filename                  = pathRadarData+PBLmode[indFile]
    # reading data 
    WbandRadarData            = Dataset(filename, mode='r')


    # reading variables 
    time                      = WbandRadarData.variables['Time'][:].copy()
    unitsTime                 = WbandRadarData.variables['Time'].Units
    milliSec                  = WbandRadarData.variables['Timems'][:].copy()
    unitsNumDate              = convertWbandTimeUnits(unitsTime)
    datetimeRadar             = nc4.num2date(WbandRadarData.variables['Time'][:],unitsNumDate)
    rangeC1                   = WbandRadarData.variables['C1Range'][:].copy()
    LWP                       = WbandRadarData.variables['LWP'][:].copy()  # in G/m^2
    CB                        = WbandRadarData.variables['CBH'][:].copy()
    ZEC1                      = WbandRadarData.variables['C1ZE'][:].copy()
    VDC1                      = WbandRadarData.variables['C1MeanVel'][:].copy()
    SWC1                      = WbandRadarData.variables['C1SpecWidth'][:].copy()
    SKC1                      = WbandRadarData.variables['C1Skew'][:].copy()    
    
    # including microseconds in the time array
    for indTime in range(len(milliSec)):
        datetimeRadar[indTime] = datetimeRadar[indTime] + timedelta(milliseconds=np.float64(milliSec[indTime]))
    
    # attaching the new file data read to the previous one
    if indFile == 0:
        timeAll                   = datetimeRadar
        LWP_arr                   = LWP
        CBH_arr                   = CB
        ZE_all                    = ZEC1
        VD_all                    = VDC1
        SW_all                    = SWC1
        SK_all                    = SKC1
    else: 
        timeAll                   = np.concatenate((timeAll, datetimeRadar))
        LWP_arr                   = np.concatenate((LWP_arr, LWP))
        CBH_arr                   = np.concatenate((CBH_arr, CB))
        ZE_all                    = np.transpose(np.column_stack((np.transpose(ZE_all), np.transpose(ZEC1))))
        VD_all                    = np.transpose(np.column_stack((np.transpose(VD_all), np.transpose(VDC1))))
        SW_all                    = np.transpose(np.column_stack((np.transpose(SW_all), np.transpose(SWC1))))
        SK_all                    = np.transpose(np.column_stack((np.transpose(SK_all), np.transpose(SKC1))))


# definition of dataframes for every variable 
timeSerie                         = pd.Series(timeAll)
LWPserie                          = pd.Series(LWP_arr, index= timeAll)
CBHserie                          = pd.Series(CBH_arr, index= timeAll)
DF_ZE                             = pd.DataFrame(ZE_all, index = timeAll, columns = rangeC1)
DF_VD                             = pd.DataFrame(VD_all, index = timeAll, columns = rangeC1)
DF_SW                             = pd.DataFrame(SW_all, index = timeAll, columns = rangeC1)
DF_SK                             = pd.DataFrame(SK_all, index = timeAll, columns = rangeC1)



# substituting -999 with np.nan in VD, SW, SK
DF_VD.values[DF_VD.values == -999.] = np.nan
DF_SW.values[DF_SW.values == -999.] = np.nan
DF_SK.values[DF_SK.values == -999.] = np.nan



# dropping duplicates in time that have been found for the high resolution mode
timeSerie                         = timeSerie.drop_duplicates(keep='first')
LWPserie                          = LWPserie[~LWPserie.index.duplicated(keep='first')]
CBHserie                          = CBHserie[~CBHserie.index.duplicated(keep='first')]
DF_ZE                             = DF_ZE[~DF_ZE.index.duplicated(keep='first')]
DF_VD                             = DF_VD[~DF_VD.index.duplicated(keep='first')]
DF_SW                             = DF_SW[~DF_SW.index.duplicated(keep='first')]
DF_SK                             = DF_SK[~DF_SK.index.duplicated(keep='first')]

# calculating ZE in dbZ
DF_ZE_db                          = pd.DataFrame(DF_ZE.values, index = timeSerie.values, columns = rangeC1)
DF_ZE_db.values[:,:]              = 10. * np.log10(DF_ZE.values[:,:])



# applying noise filter based on spectrum width 
DF_ZE_NoNoise                     = pd.DataFrame(np.full((len(timeSerie.values), len(rangeC1)), np.nan), index = timeSerie.values, columns = rangeC1)
DF_VD_NoNoise                     = pd.DataFrame(np.full((len(timeSerie.values), len(rangeC1)), np.nan), index = timeSerie.values, columns = rangeC1)
DF_SW_NoNoise                     = pd.DataFrame(np.full((len(timeSerie.values), len(rangeC1)), np.nan), index = timeSerie.values, columns = rangeC1)
DF_SK_NoNoise                     = pd.DataFrame(np.full((len(timeSerie.values), len(rangeC1)), np.nan), index = timeSerie.values, columns = rangeC1)
DF_ZE_db_NoNoise                  = pd.DataFrame(np.full((len(timeSerie.values), len(rangeC1)), np.nan), index = timeSerie.values, columns = rangeC1)

dimTime                           = len(timeSerie.values)
dimHeight                         = len(rangeC1)
for indTime in range(dimTime):
    for indHeight in range(dimHeight):
        if (DF_SW.values[indTime, indHeight] < SWNoiseThr):
            DF_ZE_NoNoise.values[indTime, indHeight] = np.nan
            DF_VD_NoNoise.values[indTime, indHeight] = np.nan
            DF_SW_NoNoise.values[indTime, indHeight] = np.nan
            DF_SK_NoNoise.values[indTime, indHeight] = np.nan
            DF_ZE_db_NoNoise.values[indTime, indHeight] = np.nan
        else:
            DF_ZE_NoNoise.values[indTime, indHeight] = DF_ZE.values[indTime, indHeight]
            DF_VD_NoNoise.values[indTime, indHeight] = DF_VD.values[indTime, indHeight]
            DF_SW_NoNoise.values[indTime, indHeight] = DF_SW.values[indTime, indHeight]
            DF_SK_NoNoise.values[indTime, indHeight] = DF_SK.values[indTime, indHeight]
            DF_ZE_db_NoNoise.values[indTime, indHeight] = DF_ZE_db.values[indTime, indHeight] 
            
            
            
# ------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------
# ------------------------------- data elaboration for plotting -------------------------------------               


# derive series of values for plotting
ZEserie                     = DF_ZE_db.values.flatten()
VDserie                     = DF_VD.values.flatten()
SWserie                     = DF_SW.values.flatten()
SKserie                     = DF_SK.values.flatten()
DistCloudTopSerie           = distCloudTop.flatten()
CloudThicknessSerie         = CloudThickness.flatten()
LWPSerie                    = LWPMatrix.flatten()

ZEserie_noNoise             = DF_ZE_db_NoNoise.values.flatten()
VDserie_noNoise             = DF_VD_NoNoise.values.flatten()
SWserie_noNoise             = DF_SW_NoNoise.values.flatten()
SKserie_noNoise             = DF_SK_NoNoise.values.flatten()


# calculate distribution of Ze for every height with noise
# calculate distributions of Ze for every height without noise
Zemin                        = -60.
Zemax                        = -10.
nbins                        = 100
binsZE                       = np.linspace(Zemin, Zemax, nbins)
binsZEplot                   = np.linspace(Zemin, Zemax, nbins-1)
DistrZEheight_noNoise        = np.zeros((nbins-1, dimHeight))
DistrZEheight                = np.zeros((nbins-1, dimHeight))
MinZEheight_noNoise          = np.zeros(dimHeight)
MinZEheight                  = np.zeros(dimHeight)

for indHeight in range(dimHeight):
    
    # reading linear Ze values with and without noise at a given height
    ZElinArrNoNoise                     = 10.**(DF_ZE_NoNoise.values[:, indHeight]/10.)
    ZElinArr                            = 10.**(DF_ZE.values[:, indHeight]/10.)
    
    # calculating distributions of ZE with and without noise for every height and storing them in the matrices
    DistrZEheight_noNoise[:, indHeight] = np.histogram(DF_ZE_db_NoNoise.values[:, indHeight], binsZE, density=False)[0]
    DistrZEheight[:, indHeight]         = np.histogram(DF_ZE.values[:, indHeight], binsZE, density=False)[0]
    
    # calculating minimum observed value of Ze for every height for noise removed ZE and normal Ze values
    MinZEheight_noNoise[indHeight]      = np.nanmin(ZElinArrNoNoise)
    MinZEheight[indHeight]              = np.nanmin(ZElinArr)
    
    

# calculating fits of order 2 to the min curves
# calculation of fit with noise not removed
MinZEheight_Fit = MinZEheight[np.where(~np.isnan(MinZEheight))]                        # removing nan values from minZE array
rangeFit        = rangeC1[np.where(~np.isnan(MinZEheight))]                            # removing corresponding heights
result_fit      = np.polyfit(rangeFit, MinZEheight_Fit, deg=2)                         # running fit  Zlin = a* height^^2
Zefit           = result_fit[2]+ result_fit[1]*rangeFit+ result_fit[0]*(rangeFit**2)   # calculating Zlin from fit
Zefit_log       = 10. * np.log10(Zefit)                                                # converting Zlin to Zdb for plotting

# calculation of fit with noise removed
MinZEheight_noNoiseFit = MinZEheight_noNoise[np.where(~np.isnan(MinZEheight_noNoise))]
MinZEheight_noNoiseFit = MinZEheight_noNoiseFit[2:80]
rangeFit_noNoise       = rangeC1[np.where(~np.isnan(MinZEheight_noNoise))]
rangeFit_noNoise       = rangeFit_noNoise[2:80]
result_fit_noNoise     = np.polyfit(rangeFit_noNoise, MinZEheight_noNoiseFit, deg=2)
Zefit_noNoise          = result_fit_noNoise[2]+ result_fit_noNoise[1]*rangeFit_noNoise+ result_fit_noNoise[0]*(rangeFit_noNoise**2)# 
Zefit_noNoise_log      = 10. * np.log10(Zefit_noNoise)

# interval of points for the fit for HR no noise: [2, 50]

# saving profiles of sensitivity in txt file for later processing
DAT_noNoise            =  np.column_stack((Zefit_noNoise, Zefit_noNoise_log, rangeFit_noNoise))
DAT                    =  np.column_stack((Zefit, Zefit_log, rangeFit))
np.savetxt(pathFig+fileOutTxt+'noNoise.txt', DAT_noNoise, delimiter=" ")
np.savetxt(pathFig+fileOutTxt+'.txt', DAT, delimiter=" ")


# ------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------
# -------------------------------  plotting -------------------------------------             





# plot ZE distributions of values 
if flagPlot_histogramSensitivity == 1:
    DistrplotNoNoise = np.ma.array(DistrZEheight_noNoise, mask=DistrZEheight_noNoise == 0)
    
    fig, ax = plt.subplots(figsize=(8,8))
    ax.spines["top"].set_visible(False) 
    ax.spines["right"].set_visible(False) 
    ax.get_xaxis().tick_bottom() 
    ax.get_yaxis().tick_left()
    cax1 = ax.pcolormesh(binsZEplot, rangeC1, DistrplotNoNoise.transpose(), vmin=0.1, vmax=20, cmap='viridis')
    #plt.plot(MinZEheight_noNoise, rangeC1, color='red', linewidth=3.)
    plt.plot(Zefit_noNoise_log, rangeFit_noNoise, color='red', linewidth=3.)
    ax.set_ylim(200.,2400.)                                               # limits of the y-axe
    ax.set_xlim(-60, -10.)                                                # limits of the x-axes
    ax.set_title("Sensitivity (noise removed)", fontsize=16)
    ax.set_xlabel("Reflectivity [dBz]", fontsize=16)
    ax.set_ylabel("height [m]", fontsize=16)
    plt.legend(loc='upper left')
    plt.grid()
    cbar = fig.colorbar(cax1, orientation='vertical')
    cbar.set_label(label="Frequencies of occurrence of Ze values ",size=14)
    cbar.ax.tick_params(labelsize=14)
    cbar.aspect=80
    fig.tight_layout()
    plt.savefig(pathFig+'sensitivity_vs_height_noNoise_'+outString+'.pdf', format='pdf')    

    Distrplot = np.ma.array(DistrZEheight, mask=DistrZEheight == 0)
    fig, ax = plt.subplots(figsize=(8,8))
    ax.spines["top"].set_visible(False) 
    ax.spines["right"].set_visible(False) 
    ax.get_xaxis().tick_bottom() 
    ax.get_yaxis().tick_left()
    cax1 = ax.pcolormesh(binsZEplot, rangeC1, Distrplot.transpose(), vmin=0.1, vmax=20, cmap='viridis')
    plt.plot(Zefit_log, rangeFit, color='red', linewidth=3.)
    ax.set_ylim(200.,2400.)                                               # limits of the y-axe
    ax.set_xlim(-60, -10.)                                                # limits of the x-axes
    ax.set_title("Sensitivity (with noise)", fontsize=16)
    ax.set_xlabel("Reflectivity [dBz]", fontsize=16)
    ax.set_ylabel("height [m]", fontsize=16)
    plt.legend(loc='upper left')
    plt.grid()
    cbar = fig.colorbar(cax1, orientation='vertical')
    cbar.set_label(label="Frequencies of occurrence of Ze values ",size=14)
    cbar.ax.tick_params(labelsize=14)
    cbar.aspect=80
    fig.tight_layout()
    plt.savefig(pathFig+'sensitivity_vs_height_'+outString+'.pdf', format='pdf')    

  
# plot distributions of ZE, VD, SW, SK values
if flagPlotDistribMomentsZE == 1:
    nbins = 100
    # plotting Ze distributions
    fig, ax   = plt.subplots(figsize=(14,6))
    ax        = plt.subplot(111)
    ax.spines["top"].set_visible(False)  
    ax.spines["right"].set_visible(False)  
    alphaval = 0.5
    ax.get_xaxis().tick_bottom()  
    ax.get_yaxis().tick_left() 
    ax.tick_params(labelsize=16)
    plt.hist(ZEserie, bins=nbins, range=[-60., -10.], \
             normed=True, color='blue', cumulative=False, alpha=alphaval, label='Noise included')
    plt.hist(ZEserie_noNoise, bins=nbins, range=[-60., -10.], \
             normed=True, color='red', cumulative=False, alpha=alphaval, label='Noise removed')
    plt.legend(loc='upper right', fontsize=12)#range=[np.nanmin(LWP_obs_PBL),np.nanmax(LWP_obs_PBL)]
    ax.set_title('distributions of Ze values', fontsize=16)
    ax.set_xlabel(' Ze [dBz]', fontsize=16)
    ax.set_ylabel('Occurrences', fontsize=16)
    plt.savefig(pathFig+'histogram_Ze_'+outString+'.pdf', format='pdf')  

if flagPlotDistribMomentsVD == 1:
    nbins = 100
    # plotting Vd distributions
    fig, ax   = plt.subplots(figsize=(14,6))
    ax        = plt.subplot(111)
    ax.spines["top"].set_visible(False)  
    ax.spines["right"].set_visible(False)  
    alphaval = 0.5
    ax.get_xaxis().tick_bottom()  
    ax.get_yaxis().tick_left() 
    ax.tick_params(labelsize=16)
    plt.hist(VDserie, bins=nbins, range=[-10., 10.], \
             normed=True, color='blue', cumulative=False, alpha=alphaval, label='Noise included')
    plt.hist(VDserie_noNoise, bins=nbins, range=[-10., 10.], \
             normed=True, color='red', cumulative=False, alpha=alphaval, label='Noise removed')
    plt.legend(loc='upper right', fontsize=12)#range=[np.nanmin(LWP_obs_PBL),np.nanmax(LWP_obs_PBL)]
    ax.set_title('distributions of Vd values', fontsize=16)
    ax.set_xlabel(' Vd [m/s]', fontsize=16)
    ax.set_ylabel('Occurrences', fontsize=16)
    plt.savefig(pathFig+'histogram_Vd_'+outString+'.pdf', format='pdf')  




if flagPlotDistribMomentsSW == 1:
    nbins = 100
    # plotting Vd distributions
    fig, ax   = plt.subplots(figsize=(14,6))
    ax        = plt.subplot(111)
    ax.spines["top"].set_visible(False)  
    ax.spines["right"].set_visible(False)  
    alphaval = 0.5
    ax.get_xaxis().tick_bottom()  
    ax.get_yaxis().tick_left() 
    ax.tick_params(labelsize=16)
    plt.hist(SWserie, bins=nbins, range=[0., 0.8], \
             normed=True, color='blue', cumulative=False, alpha=alphaval, label='Noise included')
    plt.hist(SWserie_noNoise, bins=nbins, range=[0., 0.8], \
             normed=True, color='red', cumulative=False, alpha=alphaval, label='Noise removed')
    plt.legend(loc='upper right', fontsize=12)#range=[np.nanmin(LWP_obs_PBL),np.nanmax(LWP_obs_PBL)]
    ax.set_title('distributions of Sw values', fontsize=16)
    ax.set_xlabel(' Sw [m/s]', fontsize=16)
    ax.set_ylabel('Occurrences', fontsize=16)
    plt.savefig(pathFig+'histogram_Sw_'+outString+'.pdf', format='pdf')  



if flagPlotDistribMomentsSK == 1:
    nbins = 100
    # plotting Vd distributions
    fig, ax   = plt.subplots(figsize=(14,6))
    ax        = plt.subplot(111)
    ax.spines["top"].set_visible(False)  
    ax.spines["right"].set_visible(False)  
    alphaval = 0.5
    ax.get_xaxis().tick_bottom()  
    ax.get_yaxis().tick_left() 
    ax.tick_params(labelsize=16)
    plt.hist(SKserie, bins=nbins, range=[-2., 2.], \
             normed=True, color='blue', cumulative=False, alpha=alphaval, label='Noise included')
    plt.hist(SKserie_noNoise, bins=nbins, range=[-2., 2.], \
             normed=True, color='red', cumulative=False, alpha=alphaval, label='Noise removed')
    plt.legend(loc='upper right', fontsize=12)#range=[np.nanmin(LWP_obs_PBL),np.nanmax(LWP_obs_PBL)]
    ax.set_title('distributions of Sk values', fontsize=16)
    ax.set_xlabel(' Sk [m/s]', fontsize=16)
    ax.set_ylabel('Occurrences', fontsize=16)
    plt.savefig(pathFig+'histogram_Sk_'+outString+'.pdf', format='pdf')  




#time height plot of ZE with noise
if flagplot_ZE == 1:
    # plotting matrices of the filtered moments
    ZEplot = np.ma.array(DF_ZE_db.values, mask=np.isnan(DF_ZE_db.values))
    
    fig, ax = plt.subplots(figsize=(12,6))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax.xaxis.set_minor_formatter(mdates.DateFormatter("%H:%M"))
    ax.xaxis_date()
    cax1 = ax.pcolormesh(timeSerie.values, rangeC1, ZEplot.transpose(), vmin=np.nanmin(ZEplot), vmax=np.nanmax(ZEplot), cmap='jet')
    ax.set_ylim(200.,2400.)                                               # limits of the y-axe
    ax.set_xlim()                                                        # limits of the x-axes
    ax.set_title("reflectivity 94Ghz cloud radar - BCO", fontsize=16)
    ax.set_xlabel("time [hh:mm]", fontsize=16)
    ax.set_ylabel("height [m]", fontsize=16)
    #plt.plot(time_ICON, CT_array, color='black', label='cloud top')
    #plt.plot(time_ICON, CB_array, color='black',label='cloud base')
    plt.legend(loc='upper left')
    cbar = fig.colorbar(cax1, orientation='vertical')
    #cbar.ticks=([0,1,2,3])
    #cbar.ax.set_yticklabels(['no cloud','liquid','ice','mixed phase'])
    cbar.set_label(label="Ze [dBz]",size=14)
    cbar.ax.tick_params(labelsize=14)
    cbar.aspect=80
    fig.tight_layout()
    plt.savefig(pathFig+'timeHeight_ZE_'+outString+'.png', format='png')

#time height plot of ZE with noise
if flagplot_ZEnoNoise == 1:
    ZEplotNN = np.ma.array(DF_ZE_db_NoNoise.values, mask=np.isnan(DF_ZE_db_NoNoise.values))
    
    fig, ax = plt.subplots(figsize=(12,6))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax.xaxis.set_minor_formatter(mdates.DateFormatter("%H:%M"))
    ax.xaxis_date()
    cax1 = ax.pcolormesh(timeSerie.values, rangeC1, ZEplotNN.transpose(), vmin=np.nanmin(ZEplotNN), vmax=np.nanmax(ZEplotNN), cmap='jet')
    ax.set_ylim(200.,2400.)                                               # limits of the y-axe
    ax.set_xlim()                                                        # limits of the x-axes
    ax.set_title("reflectivity 94Ghz cloud radar - BCO", fontsize=16)
    ax.set_xlabel("time [hh:mm]", fontsize=16)
    ax.set_ylabel("height [m]", fontsize=16)
    #plt.plot(time_ICON, CT_array, color='black', label='cloud top')
    #plt.plot(time_ICON, CB_array, color='black',label='cloud base')
    plt.legend(loc='upper left')
    cbar = fig.colorbar(cax1, orientation='vertical')
    #cbar.ticks=([0,1,2,3])
    #cbar.ax.set_yticklabels(['no cloud','liquid','ice','mixed phase'])
    cbar.set_label(label="Ze [dBz]",size=14)
    cbar.ax.tick_params(labelsize=14)
    cbar.aspect=80
    fig.tight_layout()
    plt.savefig(pathFig+'timeHeight_ZE_noNoise_'+outString+'.png', format='png')
    

if flagplot_VD == 1:
    VDplot = np.ma.array(DF_VD.values, mask=np.isnan(DF_VD.values))
    print('plotting Mean Doppler velocity')
    
    fig, ax = plt.subplots(figsize=(12,6))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax.xaxis.set_minor_formatter(mdates.DateFormatter("%H:%M"))
    ax.xaxis_date()
    cax1 = ax.pcolormesh(timeSerie.values, rangeC1, VDplot.transpose(), vmin=-2., vmax=2., cmap='PiYG')
    ax.set_ylim(200.,2400.)                                               # limits of the y-axe
    ax.set_xlim()                                                        # limits of the x-axes
    ax.set_title("reflectivity 94Ghz cloud radar - BCO", fontsize=16)
    ax.set_xlabel("time [hh:mm]", fontsize=16)
    ax.set_ylabel("height [m]", fontsize=16)
    #plt.plot(time_ICON, CT_array, color='black', label='cloud top')
    #plt.plot(time_ICON, CB_array, color='black',label='cloud base')
    plt.legend(loc='upper left')
    cbar = fig.colorbar(cax1, orientation='vertical')
    #cbar.ticks=([0,1,2,3])
    #cbar.ax.set_yticklabels(['no cloud','liquid','ice','mixed phase'])
    cbar.set_label(label="Vd [m/s]",size=14)
    cbar.ax.tick_params(labelsize=14)
    cbar.aspect=80
    fig.tight_layout()
    plt.savefig(pathFig+'timeHeight_VD_'+outString+'.png', format='png')

       
if flagplot_VDnoNoise == 1:
    
    VDplot = np.ma.array(DF_VD_NoNoise.values, mask=np.isnan(DF_VD_NoNoise.values))
    print('plotting Mean Doppler velocity')
    
    fig, ax = plt.subplots(figsize=(12,6))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax.xaxis.set_minor_formatter(mdates.DateFormatter("%H:%M"))
    ax.xaxis_date()
    cax1 = ax.pcolormesh(timeSerie.values, rangeC1, VDplot.transpose(), vmin=-2., vmax=2., cmap='PiYG')
    ax.set_ylim(200.,2400.)                                               # limits of the y-axe
    ax.set_xlim()                                                        # limits of the x-axes
    ax.set_title("reflectivity 94Ghz cloud radar - BCO", fontsize=16)
    ax.set_xlabel("time [hh:mm]", fontsize=16)
    ax.set_ylabel("height [m]", fontsize=16)
    #plt.plot(time_ICON, CT_array, color='black', label='cloud top')
    #plt.plot(time_ICON, CB_array, color='black',label='cloud base')
    plt.legend(loc='upper left')
    cbar = fig.colorbar(cax1, orientation='vertical')
    #cbar.ticks=([0,1,2,3])
    #cbar.ax.set_yticklabels(['no cloud','liquid','ice','mixed phase'])
    cbar.set_label(label="Vd [m/s]",size=14)
    cbar.ax.tick_params(labelsize=14)
    cbar.aspect=80
    fig.tight_layout()
    plt.savefig(pathFig+'timeHeight_VD_noNoise_'+outString+'.png', format='png')
        


if flagplot_SW == 1:
    
    SWplot = np.ma.array(DF_SW.values, mask=np.isnan(DF_SW.values))
    print('plotting Doppler spectrum width')
    
    fig, ax = plt.subplots(figsize=(12,6))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax.xaxis.set_minor_formatter(mdates.DateFormatter("%H:%M"))
    ax.xaxis_date()
    cax1 = ax.pcolormesh(timeSerie.values, rangeC1, SWplot.transpose(), vmin=0., vmax=0.7, cmap='viridis')
    ax.set_ylim(200.,2400.)                                               # limits of the y-axe
    ax.set_xlim()                                                        # limits of the x-axes
    ax.set_title("Spectrum width 94Ghz cloud radar - BCO", fontsize=16)
    ax.set_xlabel("time [hh:mm]", fontsize=16)
    ax.set_ylabel("height [m]", fontsize=16)
    #plt.plot(time_ICON, CT_array, color='black', label='cloud top')
    #plt.plot(time_ICON, CB_array, color='black',label='cloud base')
    plt.legend(loc='upper left')
    cbar = fig.colorbar(cax1, orientation='vertical')
    #cbar.ticks=([0,1,2,3])
    #cbar.ax.set_yticklabels(['no cloud','liquid','ice','mixed phase'])
    cbar.set_label(label="Sw [m/s]",size=14)
    cbar.ax.tick_params(labelsize=14)
    cbar.aspect=80
    fig.tight_layout()
    plt.savefig(pathFig+'timeHeight_SW_'+outString+'.png', format='png') 

    
if flagplot_SWnoNoise == 1:
    
    SWplot = np.ma.array(DF_SW_NoNoise.values, mask=np.isnan(DF_SW_NoNoise.values))
    print('plotting Doppler spectrum width')
    
    fig, ax = plt.subplots(figsize=(12,6))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax.xaxis.set_minor_formatter(mdates.DateFormatter("%H:%M"))
    ax.xaxis_date()
    cax1 = ax.pcolormesh(timeSerie.values, rangeC1, SWplot.transpose(), vmin=0., vmax=0.7, cmap='viridis')
    ax.set_ylim(200.,2400.)                                               # limits of the y-axe
    ax.set_xlim()                                                        # limits of the x-axes
    ax.set_title("Spectrum width 94Ghz cloud radar - BCO", fontsize=16)
    ax.set_xlabel("time [hh:mm]", fontsize=16)
    ax.set_ylabel("height [m]", fontsize=16)
    #plt.plot(time_ICON, CT_array, color='black', label='cloud top')
    #plt.plot(time_ICON, CB_array, color='black',label='cloud base')
    plt.legend(loc='upper left')
    cbar = fig.colorbar(cax1, orientation='vertical')
    #cbar.ticks=([0,1,2,3])
    #cbar.ax.set_yticklabels(['no cloud','liquid','ice','mixed phase'])
    cbar.set_label(label="Sw [m/s]",size=14)
    cbar.ax.tick_params(labelsize=14)
    cbar.aspect=80
    fig.tight_layout()
    plt.savefig(pathFig+'timeHeight_SW_noNoise_'+outString+'.png', format='png') 
        
    

if flagplot_SK == 1:
    
    SKplot = np.ma.array(DF_SK.values, mask=np.isnan(DF_SK.values))
    print('plotting Skewness')
    
    fig, ax = plt.subplots(figsize=(12,6))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax.xaxis.set_minor_formatter(mdates.DateFormatter("%H:%M"))
    ax.xaxis_date()
    cax1 = ax.pcolormesh(timeSerie.values, rangeC1, SKplot.transpose(), vmin=-1., vmax=1., cmap='viridis')
    ax.set_ylim(200.,2400.)                                               # limits of the y-axe
    ax.set_xlim()                                                        # limits of the x-axes
    ax.set_title("Skewness 94Ghz cloud radar - BCO", fontsize=16)
    ax.set_xlabel("time [hh:mm]", fontsize=16)
    ax.set_ylabel("height [m]", fontsize=16)
    #plt.plot(time_ICON, CT_array, color='black', label='cloud top')
    #plt.plot(time_ICON, CB_array, color='black',label='cloud base')
    plt.legend(loc='upper left')
    cbar = fig.colorbar(cax1, orientation='vertical')
    #cbar.ticks=([0,1,2,3])
    #cbar.ax.set_yticklabels(['no cloud','liquid','ice','mixed phase'])
    cbar.set_label(label="Sw [m/s]",size=14)
    cbar.ax.tick_params(labelsize=14)
    cbar.aspect=80
    fig.tight_layout()
    plt.savefig(pathFig+'timeHeight_SK_'+outString+'.png', format='png') 


if flagplot_SKnoNoise == 1:
    
    SKplot = np.ma.array(DF_SK_NoNoise.values, mask=np.isnan(DF_SK_NoNoise.values))
    print('plotting Skewness')
    
    fig, ax = plt.subplots(figsize=(12,6))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax.xaxis.set_minor_formatter(mdates.DateFormatter("%H:%M"))
    ax.xaxis_date()
    cax1 = ax.pcolormesh(timeSerie.values, rangeC1, SKplot.transpose(), vmin=-1., vmax=1., cmap='viridis')
    ax.set_ylim(200.,2400.)                                               # limits of the y-axe
    ax.set_xlim()                                                        # limits of the x-axes
    ax.set_title("Skewness 94Ghz cloud radar - BCO", fontsize=16)
    ax.set_xlabel("time [hh:mm]", fontsize=16)
    ax.set_ylabel("height [m]", fontsize=16)
    #plt.plot(time_ICON, CT_array, color='black', label='cloud top')
    #plt.plot(time_ICON, CB_array, color='black',label='cloud base')
    plt.legend(loc='upper left')
    cbar = fig.colorbar(cax1, orientation='vertical')
    #cbar.ticks=([0,1,2,3])
    #cbar.ax.set_yticklabels(['no cloud','liquid','ice','mixed phase'])
    cbar.set_label(label="Sw [m/s]",size=14)
    cbar.ax.tick_params(labelsize=14)
    cbar.aspect=80
    fig.tight_layout()
    plt.savefig(pathFig+'timeHeight_SK_noNoise_'+outString+'.png', format='png') 


