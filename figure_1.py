import sys
sys.path.insert(1, '/scratch/bam218/Tropopause')

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

# define the path to where the simulation output is stored
location = '/scratch/bam218/isca_data/'

control = xr.open_mfdataset(location+'1AUG2024_TROPOPAUSE_SCM_PRESCRIBE_SST_290K_take1/run001*/atmos_yearly.nc')

from __future__ import division, print_function
import numpy as np
import sys,os

sys.path.append("/home/links/bam218/PyRADS-master-brett/")
import pyrads

from scipy.integrate import trapz,simps,cumtrapz

### Helpers
class Dummy:
    pass

## setup thermodynamic parameters
params = Dummy()

params.Rv = pyrads.phys.H2O.R # moist component
params.cpv = pyrads.phys.H2O.cp
params.Lvap = pyrads.phys.H2O.L_vaporization_TriplePoint
params.satvap_T0 = pyrads.phys.H2O.TriplePointT
params.satvap_e0 = pyrads.phys.H2O.TriplePointP
params.esat = lambda T: pyrads.Thermodynamics.get_satvps(T,params.satvap_T0,params.satvap_e0,params.Rv,params.Lvap)

params.R = pyrads.phys.air.R  # dry component
params.cp = pyrads.phys.air.cp
params.R_CO2 = pyrads.phys.CO2.R
params.ps_dry = 1e5           # surface pressure of dry component

params.g = 9.8             # surface gravity
params.cosThetaBar = 3./5. # average zenith angle used in 2stream eqns
params.RH = 1.             # relative humidity


## setup resolution (vertical,spectral)

N_press = 81      # for testing only!
# dwavenr = 0.1     #  for testing only!

#N_press = 60       #
wavenr_min = 0.1   # [cm^-1]
wavenr_max = 3500. #
dwavenr = 0.1     #

Tstrat = 150.      # stratospheric temperature
## setup range of temperatures, and if/where output is saved to:
Ts = 290.
ppv_CO2 = 0
# ppv_CO2 = 280e-6


# Now let's try to compute the profile of cooling for Isca's SCM control atmosphere
location = '/scratch/bam218/isca_data/'

control = xr.open_mfdataset(location+'1AUG2024_TROPOPAUSE_SCM_PRESCRIBE_SST_290K_take1/run001*/atmos_yearly.nc')

scm = control


# convert simulation output into shape required for PyRADS
scm_T = np.ndarray.flatten(scm.temp.values)
scm_Ts = np.ndarray.flatten(scm.t_surf.values)[0]
scm_p = np.ndarray.flatten(scm.pfull.values)*100 # convert to Pa
scm_ps = np.ndarray.flatten(scm.ps.values)[0]
scm_q = np.ndarray.flatten(scm.sphum.values)
scm_Tstrat = scm_T[0]

scm_T = np.append(scm_T, scm_Ts)
scm_p = np.append(scm_p, scm_ps)
scm_T = scm_T.astype(np.float128)
scm_Ts = scm_Ts.astype(np.float128)
scm_p = scm_p.astype(np.float128)
scm_ps = scm_ps.astype(np.float128)
scm_q = scm_q.astype(np.float128)
scm_Tstrat = scm_Tstrat.astype(np.float128)

g = pyrads.SetupGrids.make_grid( Ts,Tstrat,N_press,wavenr_min,wavenr_max,dwavenr,params, RH=1)

g.T = scm_T
g.p = scm_p
g.q = scm_q
g.Ts = scm_Ts

## MAIN LOOP
print( "wavenr_min,wavenr_max,dwave [cm^-1] = %.4f,%.4f,%.4f" % (wavenr_min,wavenr_max,dwavenr))
print( "\n")
print( "N_press = %.1f" % N_press)
print( "\n")
print( "Surface temperature = %.1f K" % Ts)

# # setup grid:
# g = pyrads.SetupGrids.make_grid( Ts,Tstrat,N_press,wavenr_min,wavenr_max,dwavenr,params, RH=1)

g.tau = pyrads.OpticalThickness.compute_tau_H2ON2_CO2dilute(g.p,g.T,g.q,ppv_CO2,g,params,RH=1)

# compute Planck functions etc:
#   -> here: fully spectrally resolved!
T_2D = np.tile( g.T, (g.Nn,1) ).T               # [press x wave]
g.B_surf = np.pi* pyrads.Planck.Planck_n( g.n,g.Ts )     # [wave]
g.B = np.pi* pyrads.Planck.Planck_n( g.wave, T_2D )    # [press x wave]

# compute OLR etc:
olr_spec = pyrads.Get_Fluxes.Fplus_alternative(0,g) # (spectrally resolved=irradiance)
olr = simps(olr_spec,g.n)

print( "OLR = ",olr)

olr_nu = pyrads.Get_Fluxes.Fplus_alternative(0,g)
flux_nu = np.zeros([len(g.p),len(olr_nu)])
new_shape = (len(g.p), len(olr_nu))
p = np.tile(g.p[:, np.newaxis], new_shape[1])
for i in range(len(g.p)):
    flux_nu_i = pyrads.Get_Fluxes.Fplus_alternative(i,g) # spectral flux at level i
    flux_nu[i,:] = flux_nu_i
        
dFnu_dp =  np.gradient(flux_nu, axis=0)/np.gradient(p, axis=0) 
Q_nu = -dFnu_dp * params.g / params.cp * 86400 # K day^-1 cm
tau_nu = g.tau

Q = simps(Q_nu,g.n)

wavenr_min = 0.1   # [cm^-1]
wavenr_max = 3500. #
dwavenr = 0.1

nu = np.linspace(wavenr_min,wavenr_max,34999)
nu_coarse = np.linspace(wavenr_min,wavenr_max,3499)

from matplotlib import ticker
tick_locator = ticker.MaxNLocator(nbins=3)

fig, axes, cax = fc(1,1,width=2.,aspect=1,right_pad=0,cbar_mode='each',cbar_pad=0.6,cbar_location='bottom',cbar_short_side_pad=0)

ax = axes[0]
cax = cax[0]

c = ax.contourf(nu, g.p/100, Q_nu, add_colorbar=False, vmin=-0.003, vmax=-0.0001, cmap='inferno_r', levels=np.linspace(-0.003,-0.0001,100), extend='both')

ax.set_title('')
ax.set_xlabel('Wavenumber / cm$^{-1}$')
ax.set_ylabel('Pressure / hPa')
ax.set_xlim([0,1500])
ax.set_ylim([1000,0])
cb = plt.colorbar(c, cax=cax, label=r'K day$^{-1}$ cm', orientation='horizontal')
# cb.cmap.set_under('white')
# cb.cmap.set_over('whitesmoke')
cb.cmap.set_over('black')
cb.locator = tick_locator
cb.update_ticks()

from matplotlib import ticker
tick_locator = ticker.MaxNLocator(nbins=3)

fig, axes = fc(1,1,width=1.4,aspect=1.4,right_pad=0)
ax = axes[0]

ppf.make_pretty_plot(ax, xmin=-3, xmax=3, ymin=1000, ymax=0, xlabel=r'Radiative heating / K day$^{-1}$',ylabel=r'Pressure / hPa', delxmaj=3,delxmin=0.5,delymaj=200,delymin=200)

ax.axvline(x=0, color='silver', alpha=0.75, linewidth=0.5)
ax.plot(Q,g.p/100, color='cornflowerblue',linewidth=1)
ax.fill_betweenx(g.p/100, Q, zorder=9, color='cornflowerblue',alpha=0.15)

#  compare #26428b to 'navy'
mpl.rcParams['figure.dpi']= 300
fig, ax = fc(1, 1, width=2, aspect=1)
ax = ax[0]
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.tick_params(axis='both', direction='out')
ax.set_ylim([1e-6, 1e7])
ax.set_xlim([0,1500])
ax.set_xlabel(r'$\nu$ (cm$^{-1}$)')
ax.set_ylabel(r'$\kappa$ (m$^2$ kg$^{-1}$)')
ax.set_yscale('log', basey=10)

pressures=[500]
countblue=0
for pressure in pressures:
    countblue+=1
    wavenr_min = 0.1   # [cm^-1]
    wavenr_max = 3500. #
    dwavenr = 0.01     #
    wavenums = np.linspace(wavenr_min, wavenr_max, (wavenr_max-wavenr_min)/dwavenr+1)
    kappash2o = pyrads.Absorption_Crosssections_HITRAN2016.getKappa_HITRAN(waveGrid=wavenums,wave0=wavenr_min,
                                                                       wave1=wavenr_max,delta_wave=dwavenr,
                                                                       molecule_name='H2O',press=pressure*100,temp=260.,
                                                                       lineWid=25.,broadening="mixed",
                                                                       press_self=150, cutoff_option="fixed",
                                                                       remove_plinth=False)

    kap_h2o_rot = kappash2o[wavenums<=1000]
    kap_h2o_vr  = kappash2o[wavenums>=1000]

    kap_xr = xr.DataArray(data=kappash2o, coords={'nu': wavenums}, dims=('nu'), name='kap', attrs={'created by': 'rac_cooling.ipynb'})
    ax.scatter(kap_xr.nu, kap_xr, color='cornflowerblue', s=0.015, alpha=0.035)

fig, ax = fc(1, 1, width=1, aspect=3)
ax = ax[0]
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['left'].set_visible(True)
ax.tick_params(axis='both', direction='out')  
ax.set_yscale('log', basey=10)
ax.set_xlim([-0.01,0.11])
ax.set_ylim([1e-6,1e7])
ax.set_xlabel(r'PDF')
ax.set_ylabel(r'$\kappa$ (m$^2$ kg$^{-1}$)')

countblue=0
for pressure in pressures:
    countblue+=1
    wavenr_min = 0.1   # [cm^-1]
    wavenr_max = 3500. #
    dwavenr = 0.01     #
    wavenums = np.linspace(wavenr_min, wavenr_max, (wavenr_max-wavenr_min)/dwavenr+1)
    kappash2o = pyrads.Absorption_Crosssections_HITRAN2016.getKappa_HITRAN(waveGrid=wavenums,wave0=wavenr_min,
                                                                       wave1=wavenr_max,delta_wave=dwavenr,
                                                                       molecule_name='H2O',press=pressure*100,temp=260.,
                                                                       lineWid=25.,broadening="mixed",
                                                                       press_self=150, cutoff_option="fixed",
                                                                       remove_plinth=False)

    kap_h2o_rot = kappash2o[wavenums<=1000]
    kap_h2o_vr  = kappash2o[wavenums>=1000]
    numbins=150
    hist_kap_rot, bin_edges = np.histogram(np.log(kap_h2o_rot), bins=numbins, range=None, normed=None, weights=None, density=True)

    ax.plot(hist_kap_rot, np.exp(bin_edges[:-1]), linewidth=0.5, color='cornflowerblue')
    ax.fill_between(hist_kap_rot, np.exp(bin_edges[:-1]), zorder=10, color= 'cornflowerblue', alpha=0.15)
    ax.scatter(0.0944, 40, color='k', s=2, zorder=10)


