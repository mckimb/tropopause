import sys
sys.path.insert(1, '/scratch/bam218/Tropopause')

%matplotlib inline
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib as mpl
import analyze_tropopause_20JAN as at
from matplotlib import cm
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import LogFormatter 
from faceted import faceted as fc
from scipy.integrate import trapz,simps,cumtrapz
import trop_constants as tc
import pretty_plotting_20JAN as ppf
import warnings
import scipy.special.lambertw as lambertw
import shared_tropopause_funcs as tp
warnings.filterwarnings('ignore')

mpl.rcParams['text.usetex'] = True
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = ['Computer Modern Roman']
mpl.rcParams['text.usetex'] = True
mpl.rcParams['figure.dpi']= 300
#----------------------------------------

location = '/scratch/bam218/isca_data/'

###########################################################
############## SPECTRAL MOIST SST RUNS ####################
###########################################################
col270 = xr.open_mfdataset(location+'1AUG2024_TROPOPAUSE_SCM_PRESCRIBE_SST_270K_take1/run001*/atmos_yearly.nc')
col280 = xr.open_mfdataset(location+'1AUG2024_TROPOPAUSE_SCM_PRESCRIBE_SST_280K_take1/run001*/atmos_yearly.nc')
col290 = xr.open_mfdataset(location+'1AUG2024_TROPOPAUSE_SCM_PRESCRIBE_SST_290K_take1/run001*/atmos_yearly.nc')
col300 = xr.open_mfdataset(location+'1AUG2024_TROPOPAUSE_SCM_PRESCRIBE_SST_300K_take1/run001*/atmos_yearly.nc')
col310 = xr.open_mfdataset(location+'1AUG2024_TROPOPAUSE_SCM_PRESCRIBE_SST_310K_take1/run001*/atmos_yearly.nc')

data_list = [col270, col280, col290, col300, col310]

####################################
###### PRESCRIBE_SST GRAY DRY ######
####################################
prescribe_sst_270 = xr.open_mfdataset(location+'31JUL2023_TROPOPAUSE_SCM_GRAY_PRESCRIBE_SST_270K_take3/run0010/atmos_yearly.nc', decode_times=False)
prescribe_sst_280 = xr.open_mfdataset(location+'31JUL2023_TROPOPAUSE_SCM_GRAY_PRESCRIBE_SST_280K_take3/run0010/atmos_yearly.nc', decode_times=False)
prescribe_sst_290 = xr.open_mfdataset(location+'31JUL2023_TROPOPAUSE_SCM_GRAY_PRESCRIBE_SST_290K_take3/run0010/atmos_yearly.nc', decode_times=False)
prescribe_sst_300 = xr.open_mfdataset(location+'31JUL2023_TROPOPAUSE_SCM_GRAY_PRESCRIBE_SST_300K_take3/run0010/atmos_yearly.nc', decode_times=False)
prescribe_sst_310 = xr.open_mfdataset(location+'31JUL2023_TROPOPAUSE_SCM_GRAY_PRESCRIBE_SST_310K_take3/run0010/atmos_yearly.nc', decode_times=False)
gray_data_list = [prescribe_sst_270,prescribe_sst_280,prescribe_sst_290,prescribe_sst_300,prescribe_sst_310]

######################################
###### PRESCRIBE_SST GRAY MOIST ######
######################################
prescribe_sst_270 = xr.open_mfdataset(location+'22AUG2023_TROPOPAUSE_SCM_GRAYMOISTPRESCRIBE_SST_270K_take1/run0010/atmos_yearly.nc', decode_times=False)
prescribe_sst_280 = xr.open_mfdataset(location+'22AUG2023_TROPOPAUSE_SCM_GRAYMOISTPRESCRIBE_SST_280K_take1/run0010/atmos_yearly.nc', decode_times=False)
prescribe_sst_290 = xr.open_mfdataset(location+'22AUG2023_TROPOPAUSE_SCM_GRAYMOISTPRESCRIBE_SST_290K_take1/run0010/atmos_yearly.nc', decode_times=False)
prescribe_sst_300 = xr.open_mfdataset(location+'22AUG2023_TROPOPAUSE_SCM_GRAYMOISTPRESCRIBE_SST_300K_take1/run0010/atmos_yearly.nc', decode_times=False)
prescribe_sst_310 = xr.open_mfdataset(location+'22AUG2023_TROPOPAUSE_SCM_GRAYMOISTPRESCRIBE_SST_310K_take1/run0010/atmos_yearly.nc', decode_times=False)
gray_moist_data_list = [prescribe_sst_270,prescribe_sst_280,prescribe_sst_290,prescribe_sst_300,prescribe_sst_310]

##############################
### TESTING OLR CONSTRAINT ###
##################a###########

width=2
sigma=5.67e-8

###############################
####### SPECTRAL MODELS #######
###############################
fig, axes = fc(1, 1, width=width, aspect=1, internal_pad=0)
ax = axes[0]
ppf.make_pretty_plot(ax, xmin=260, xmax=320, ymin=160, ymax=260, xlabel=r'$T_s$ / K',ylabel=r'$T_{tp}$ / K', delymaj=20,delymin=5, delxmaj=20, delxmin=5)

ttp_list = []
ts_list = []
olr_list = []
ttp_predic_list = []
ttp_predic_list2 = []
count=0
for data in data_list:
    count+=1
    tsurf_bin = data.t_surf.mean(('lon','lat','time'))
    olr_bin = data.olr.mean().values
    olr_list.append(olr_bin)
    ttp_bin, ptp, ztp = tp.get_tropopause_temp(data,threshold=-0.05,cutoff=0)
    ttp_list.append(ttp_bin)
    ts_list.append(tsurf_bin.values)
    ax.scatter(tsurf_bin, ttp_bin, color='lightsteelblue', edgecolor='navy',s=15, zorder=1,linewidths=0.5)
    ax.scatter(tsurf_bin, (olr_bin/(2*sigma))**(1/4), color='lightsteelblue', edgecolor='navy',s=15,zorder=1,linewidths=0.5, marker='s')

###############################
###### GRAY DRY MODELS ########
###############################
fig, axes = fc(1, 1, width=width, aspect=1, internal_pad=0)
ax = axes[0]
ppf.make_pretty_plot(ax, xmin=260, xmax=320, ymin=160, ymax=260, xlabel=r'$T_s$ / K',ylabel=r'$T_{tp}$ / K', delymaj=20,delymin=5, delxmaj=20, delxmin=5)

ttp_list = []
ts_list = []
olr_list = []
ttp_predic_list = []
ttp_predic_list2 = []
count=0
for data in gray_data_list:
    count+=1
    olr_bin = data.olr.mean().values
    olr_list.append(olr_bin)
    tsurf_bin = data.t_surf.mean(('lon','lat','time'))
    ttp_bin, ptp, ztp = tp.get_tropopause_temp(data,threshold=-0.05,cutoff=0)
    ttp_list.append(ttp_bin)
    ts_list.append(tsurf_bin.values)
    ax.scatter(tsurf_bin, ttp_bin, color='lightgoldenrodyellow', edgecolor='darkgoldenrod',s=15, zorder=1,linewidths=0.5)
    ax.scatter(tsurf_bin, (olr_bin/(2*sigma))**(1/4), color='lightgoldenrodyellow', edgecolor='darkgoldenrod',s=15,zorder=1,linewidths=0.5, marker='s')

###############################
###### GRAY MOIST MODELS ######
###############################
fig, axes = fc(1, 1, width=width, aspect=1, internal_pad=0)
ax = axes[0]
ppf.make_pretty_plot(ax, xmin=260, xmax=320, ymin=160, ymax=260, xlabel=r'$T_s$ / K',ylabel=r'$T_{tp}$ / K', delymaj=20,delymin=5, delxmaj=20, delxmin=5)

ttp_list = []
ts_list = []
olr_list = []
ttp_predic_list = []
ttp_predic_list2 = []
count=0
for data in gray_moist_data_list:
    count+=1
    olr_bin = data.olr.mean().values
    olr_list.append(olr_bin)
    tsurf_bin = data.t_surf.mean(('lon','lat','time'))
    ttp_bin, ptp, ztp = tp.get_tropopause_temp(data,threshold=-0.05,cutoff=0)
    ttp_list.append(ttp_bin)
    ts_list.append(tsurf_bin.values)
    ax.scatter(tsurf_bin, ttp_bin, color='lightcyan', edgecolor='darkcyan',s=15, zorder=1,linewidths=0.5)
    ax.scatter(tsurf_bin, (olr_bin/(2*sigma))**(1/4), color='lightcyan', edgecolor='darkcyan',s=15,zorder=1,linewidths=0.5, marker='s')

tp.plot_radcool_scm_sst(data_list=data_list, threshold=-0.05, width=1.5, cutoff=0,spectral=True)

tp.plot_radcool_scm_sst(data_list=gray_data_list, threshold=-0.05, width=1.5, cutoff=0,spectral=False)

tp.plot_radcool_scm_sst(data_list=gray_moist_data_list, threshold=-0.05, width=1.5, cutoff=0,spectral=False)
