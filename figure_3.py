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
from matplotlib import ticker
tick_locator = ticker.MaxNLocator(nbins=3)
import scipy.special.lambertw as lambertw
import shared_tropopause_funcs as tp
warnings.filterwarnings('ignore')

mpl.rcParams['text.usetex'] = True
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = ['Computer Modern Roman']
mpl.rcParams['text.usetex'] = True
mpl.rcParams['figure.dpi']= 300
#----------------------------------------

# define the path to where the simulation output is stored
location = '/scratch/bam218/isca_data/'

# load in the gcm experiments where kappa is modified
kap1div2 = xr.open_mfdataset(location+'20JAN2023_TROPOPAUSE_APE_0K_KAPPAx1div2_take_1/run0025/atmos_yearly.nc', decode_times=False)
kap1 = xr.open_mfdataset(location+'14DEC2022_TROPOPAUSE_APE_0K_take_1/run0025/atmos_yearly.nc', decode_times=False)
kap2 = xr.open_mfdataset(location+'7JAN2023_TROPOPAUSE_APE_0K_KAPPAx4_take_1/run0025/atmos_yearly.nc', decode_times=False) # note the mislabeled x4 name. Should be KAPPAx2
data_list = [kap1div2, kap1, kap2]

fig, axes, cax = fc(1,1,width=2.5,aspect=1/1.4,right_pad=0,cbar_mode='each',cbar_pad=0.1,cbar_location='right',cbar_short_side_pad=0)

ax = axes[0]
cax = cax[0]

temp = kap1.temp.mean(('time','lon'))

c = ax.contourf(temp.lat.values, temp.pfull.values, temp.values, add_colorbar=False, vmin=150, vmax=250, cmap='Oranges', levels=np.linspace(150,250,11), extend='both')
contours = ax.contour(radcool.lat.values, radcool.pfull.values, radcool.values, levels=[-0.4,-0.3,-0.2,-0.1], extend='both', colors='k', linewidths=0.1,linestyles='solid')
ax.clabel(contours, inline=1, fmt='%.1f', colors='k',fontsize='x-small')
ax.tick_params(axis='both', which='both', direction='in')
ax.set_title('')
ax.set_xlabel('Latitude')
ax.set_ylabel('Pressure')
ax.set_xticks([-90,-45,0,45,90])
ax.set_ylim([300,0])
cb = plt.colorbar(c, cax=cax, label='K')
cb.locator = tick_locator
cb.update_ticks()

from matplotlib import ticker
tick_locator = ticker.MaxNLocator(nbins=3)

fig, axes, cax = fc(1,1,width=2.5,aspect=1/1.4,right_pad=0,cbar_mode='each',cbar_pad=0.1,cbar_location='right',cbar_short_side_pad=0)

ax = axes[0]
cax = cax[0]

radcool = kap1.tdt_rad.mean(('time','lon')) * 86400

c = ax.contourf(radcool.lat.values, radcool.pfull.values, radcool.values, add_colorbar=False, vmin=-1, vmax=0, cmap='Greys_r', levels=np.linspace(-1,0,11), extend='both')

contours = ax.contour(radcool.lat.values, radcool.pfull.values, radcool.values, levels=[-0.4,-0.3,-0.2,-0.1], extend='both', colors='k', linewidths=0.1,linestyles='solid')
ax.clabel(contours, inline=1, fmt='%.1f', colors='k',fontsize='x-small')
ax.tick_params(axis='both', which='both', direction='in')

ax.set_title('')
ax.set_xlabel('Latitude')
ax.set_ylabel('Pressure')
ax.set_xticks([-90,-45,0,45,90])
ax.set_ylim([300,0])
cb = plt.colorbar(c, cax=cax, label=r'K day$^{-1}$')
cb.locator = tick_locator
cb.update_ticks()

experiment = kap1

p = experiment.pfull * 100
T = experiment.temp.mean(('time','lon'))
dT_dp = T.differentiate('pfull') / 100
lapse_rate = p * tc.g * dT_dp / (tc.Rd * T) * 1000
dry_lapse_rate = 9.8

radiative_cooling = experiment.tdt_rad.mean(('time','lon'))
stability = tc.Rd/tc.cp * T/p * (1-lapse_rate/dry_lapse_rate)

mass_flux = -radiative_cooling/stability / 100 * 86400 # hPa/day

from matplotlib import ticker
tick_locator = ticker.MaxNLocator(nbins=3)

fig, axes, cax = fc(1,1,width=2.5,aspect=1/1.4,right_pad=0,cbar_mode='each',cbar_pad=0.1,cbar_location='right',cbar_short_side_pad=0)

ax = axes[0]
cax = cax[0]

c = ax.contourf(mass_flux.lat.values, mass_flux.pfull.values, mass_flux.values, add_colorbar=False, vmin=0, vmax=50, cmap='PuBu', levels=np.linspace(0,50,11), extend='both')

contours = ax.contour(radcool.lat.values, radcool.pfull.values, radcool.values, levels=[-0.4,-0.3,-0.2,-0.1], extend='both', colors='k', linewidths=0.1,linestyles='solid')
ax.clabel(contours, inline=1, fmt='%.1f', colors='k',fontsize='x-small')
ax.tick_params(axis='both', which='both', direction='in')
ax.set_title('')
ax.set_xlabel('Latitude')
ax.set_ylabel('Pressure')
ax.set_xticks([-90,-45,0,45,90])
ax.set_ylim([300,0])
cb = plt.colorbar(c, cax=cax, label=r'hPa day$^{-1}$')
cb.locator = tick_locator
cb.update_ticks()

average='none'
scm=False
width=2

tp.plot_rad_tropopause_zonal(data_list=data_list, threshold=-0.2, width=2, cutoff=0, vertical_coordinate='foo')

gamma_threshold = 5
tp.plot_lapse_tropopause_zonal(data_list,threshold=gamma_threshold,average=average,width=width,scm=scm,vertical_coordinate="temperature")
tp.plot_lapse_tropopause_zonal(data_list,threshold=gamma_threshold,average=average,width=width,scm=scm,vertical_coordinate="pressure")
