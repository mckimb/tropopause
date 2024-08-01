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

# define the path to where the simulation output is stored
location = '/scratch/bam218/isca_data/'

# load in the single column model control simulation at Ts=290, RH=0.7, kappa=1
col_sst_290 = xr.open_mfdataset(location+'1AUG2024_TROPOPAUSE_SCM_PRESCRIBE_SST_290K_take1/run001*/atmos_yearly.nc')

tp.plot_qv_scm(data=col_sst_290, threshold=-0.05, width=1.5, cutoff=0)
tp.plot_temp_scm(data=col_sst_290, threshold=-0.05, width=1.5, cutoff=0)
tp.plot_rh_scm(data=col_sst_290, threshold=-0.05, width=1.5, cutoff=0)
tp.plot_radcool_scm(data=col_sst_290, threshold=-0.05, width=1.5, cutoff=0)
tp.plot_lapse_scm(data=col_sst_290, threshold=5, width=1.5, cutoff=0)

###########################################################
###################### SST RUNS ###########################
###########################################################
col265 = xr.open_mfdataset(location+'1AUG2024_TROPOPAUSE_SCM_PRESCRIBE_SST_265K_take1/run001*/atmos_yearly.nc')
col270 = xr.open_mfdataset(location+'1AUG2024_TROPOPAUSE_SCM_PRESCRIBE_SST_270K_take1/run001*/atmos_yearly.nc')
col275 = xr.open_mfdataset(location+'1AUG2024_TROPOPAUSE_SCM_PRESCRIBE_SST_275K_take1/run001*/atmos_yearly.nc')
col280 = xr.open_mfdataset(location+'1AUG2024_TROPOPAUSE_SCM_PRESCRIBE_SST_280K_take1/run001*/atmos_yearly.nc')
col285 = xr.open_mfdataset(location+'1AUG2024_TROPOPAUSE_SCM_PRESCRIBE_SST_285K_take1/run001*/atmos_yearly.nc')
col290 = xr.open_mfdataset(location+'1AUG2024_TROPOPAUSE_SCM_PRESCRIBE_SST_290K_take1/run001*/atmos_yearly.nc')
col295 = xr.open_mfdataset(location+'1AUG2024_TROPOPAUSE_SCM_PRESCRIBE_SST_295K_take1/run001*/atmos_yearly.nc')
col300 = xr.open_mfdataset(location+'1AUG2024_TROPOPAUSE_SCM_PRESCRIBE_SST_300K_take1/run001*/atmos_yearly.nc')
col305 = xr.open_mfdataset(location+'1AUG2024_TROPOPAUSE_SCM_PRESCRIBE_SST_305K_take1/run001*/atmos_yearly.nc')
col310 = xr.open_mfdataset(location+'1AUG2024_TROPOPAUSE_SCM_PRESCRIBE_SST_310K_take1/run001*/atmos_yearly.nc')
col315 = xr.open_mfdataset(location+'1AUG2024_TROPOPAUSE_SCM_PRESCRIBE_SST_315K_take1/run001*/atmos_yearly.nc')

data_list = [col265, col270, col275, col280, col285, col290, col295, col300, col305, col310, col315]

##############################
##### Ttp VS SST ##########
##################a############
ttp_list = []
ts_list = []
olr_list = []
ttp_predic_list = []
ttp_predic_list2 = []
width=2
sigma=5.67e-8

kaptp=5500

fig, axes = fc(1, 1, width=width, aspect=1, internal_pad=0)
ax = axes[0]
ppf.make_pretty_plot(ax, xmin=260, xmax=320, ymin=150, ymax=210, xlabel=r'$T_s$ / K',ylabel=r'$T_{tp}$ / K', delymaj=20,delymin=5, delxmaj=20, delxmin=5)

count=0
for data in data_list:
    count+=1
    tsurf_bin = data.t_surf.mean(('lon','lat','time'))
    olr_list.append(data.olr.mean().values)
    ttp_bin, ptp, ztp = tp.get_tropopause_temp(data,threshold=-0.05,cutoff=0)
    ttp_list.append(ttp_bin)
    ts_list.append(tsurf_bin.values)
    ttp_predic = tp.ssm_tp(RH=0.7, kaptp=kaptp, data=data)
    ttp_predic_list.append(ttp_predic)
    ax.scatter(tsurf_bin, ttp_bin, color='lightsteelblue', edgecolor='navy',s=15, zorder=1,linewidths=0.5)
    ax.plot(ts_list, np.array(ts_list)/7 + 138, color='lightgray',linestyle='dashed', linewidth=0.9)
    ax.text(278,205,r'$\kappa_{tune}$ = '+str(int(kaptp))+' m$^2$/kg', color='navy', fontsize='x-small')
    ax.plot(ts_list, ttp_predic_list, color='navy', linewidth=1,zorder=0)

##########################
###### KAPPA RUNS ########
##########################
col1div8 = xr.open_mfdataset(location+'production_col_rrtm_kappaexpx1div8_290K_take1/run002*/atmos_yearly.nc')
col1div4 = xr.open_mfdataset(location+'production_col_rrtm_kappaexpx1div4_290K_take1/run002*/atmos_yearly.nc')
col1div2 = xr.open_mfdataset(location+'production_col_rrtm_kappaexpx1div2_290K_take1/run002*/atmos_yearly.nc')
col1 = xr.open_mfdataset(location+'production_col_rrtm_kappaexpx1_290K_take1/run002*/atmos_yearly.nc')
col2 = xr.open_mfdataset(location+'production_col_rrtm_kappaexpx2_290K_take1/run002*/atmos_yearly.nc')
col4 = xr.open_mfdataset(location+'production_col_rrtm_kappaexpx4_290K_take1/run002*/atmos_yearly.nc')
col8 = xr.open_mfdataset(location+'production_col_rrtm_kappaexpx8_290K_take3/run002*/atmos_yearly.nc')
data_list = [col1div8, col1div4, col1div2, col1, col2, col4, col8]

##############################
##### Ttp VS KAPPA ##########
##################a############
ttp_list = []
ts_list = []
olr_list = []
ttp_predic_list = []
ttp_predic_list2 = []
width=2

kaptp=5500
scales = np.array([1/8,1/4,1/2,1,2,4,8])

fig, axes = fc(1, 1, width=width, aspect=1, internal_pad=0)
ax = axes[0]
ppf.make_pretty_plot(ax, xmin=1/16, xmax=16, ymin=150, ymax=210, xlabel=r'$\kappa$ scaling',ylabel=r'$T_{tp}$ / K', xscalelog=True, delymaj=20,delymin=5, delxmaj=2)

plt.minorticks_off()

ax.set_xticks([1/16,1/8,1/4,1/2,1,2,4,8,16])
ax.set_xticklabels([r'$\frac{1}{16}$',r'$\frac{1}{8}$',r'$\frac{1}{4}$',r'$\frac{1}{2}$','1','2','4','8','16'])

count=0
for data in data_list:
    count+=1
    tsurf_bin = data.t_surf.mean(('lon','lat','time'))
    ttp_bin, ptp, ztp = tp.get_tropopause_temp(data,threshold=-0.05,cutoff=0)
    ttp_list.append(ttp_bin)
    ts_list.append(tsurf_bin.values)
    ttp_predic = tp.ssm_tp(RH=0.7, kaptp=kaptp*scales[count-1], data=data)
    ttp_predic_list.append(ttp_predic)
    ax.scatter(scales[count-1], ttp_bin, color='lightsteelblue', edgecolor='navy',s=15, zorder=1,linewidths=0.5)

ax.plot(scales, ttp_predic_list, color='navy', linewidth=1,zorder=0)
ax.plot(scales,np.array([190,186,182,178,174,170,166])-2, color='lightgray', linewidth=0.9,zorder=0, linestyle='dashed')

##########################
######### RH RUNS ########
##########################
col10 = xr.open_mfdataset(location+'1AUG2024_TROPOPAUSE_SCM_PRESCRIBE_RH_0pt1_take1/run001*/atmos_yearly.nc')
col20 = xr.open_mfdataset(location+'1AUG2024_TROPOPAUSE_SCM_PRESCRIBE_RH_0pt2_take1/run001*/atmos_yearly.nc')
col30 = xr.open_mfdataset(location+'1AUG2024_TROPOPAUSE_SCM_PRESCRIBE_RH_0pt3_take1/run001*/atmos_yearly.nc')
col40 = xr.open_mfdataset(location+'1AUG2024_TROPOPAUSE_SCM_PRESCRIBE_RH_0pt4_take1/run001*/atmos_yearly.nc')
col50 = xr.open_mfdataset(location+'1AUG2024_TROPOPAUSE_SCM_PRESCRIBE_RH_0pt5_take1/run001*/atmos_yearly.nc')
col60 = xr.open_mfdataset(location+'1AUG2024_TROPOPAUSE_SCM_PRESCRIBE_RH_0pt6_take1/run001*/atmos_yearly.nc')
col70 = xr.open_mfdataset(location+'1AUG2024_TROPOPAUSE_SCM_PRESCRIBE_RH_0pt7_take1/run001*/atmos_yearly.nc')
col80 = xr.open_mfdataset(location+'1AUG2024_TROPOPAUSE_SCM_PRESCRIBE_RH_0pt8_take1/run001*/atmos_yearly.nc')
col90 = xr.open_mfdataset(location+'1AUG2024_TROPOPAUSE_SCM_PRESCRIBE_RH_0pt9_take1/run001*/atmos_yearly.nc')
data_list = [col10, col20, col30, col40, col50, col60, col70, col80, col90]

##############################
##### Ttp VS RH ##########
##################a############
ttp_list = []
ts_list = []
olr_list = []
ttp_predic_list = []
ttp_predic_list2 = []
width=2

kaptp=5500

fig, axes = fc(1, 1, width=width, aspect=1, internal_pad=0)
ax = axes[0]
ppf.make_pretty_plot(ax, xmin=0, xmax=100, ymin=150, ymax=210, xlabel=r'RH / \%',ylabel=r'$T_{tp}$ / K', delymaj=20,delymin=5, delxmaj=20, delxmin=10)

rh_list = np.array([10,20,30,40,50,60,70,80,90])

count=0
for data in data_list:
    count+=1
    tsurf_bin = data.t_surf.mean(('lon','lat','time'))
    ttp_bin, ptp, ztp = tp.get_tropopause_temp(data,threshold=-0.05,cutoff=0)
    ttp_list.append(ttp_bin)
    ts_list.append(tsurf_bin.values)
    ttp_predic = tp.ssm_tp(RH=rh_list[count-1]/100, kaptp=kaptp, data=data)
    ttp_predic_list.append(ttp_predic)
    ax.scatter(rh_list[count-1], ttp_bin, color='lightsteelblue', edgecolor='navy',s=15, zorder=1,linewidths=0.5)

ax.plot(rh_list, ttp_predic_list, color='navy', linewidth=1,zorder=0)
