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
warnings.filterwarnings('ignore')

mpl.rcParams['text.usetex'] = True
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = ['Computer Modern Roman']
mpl.rcParams['text.usetex'] = True
mpl.rcParams['figure.dpi']= 300
#----------------------------------------

def get_tropopause_temp(data, threshold,cutoff):
    '''
    Get the tropopause temperature from using the radiative cooling profile       (K/day)
    '''
    
    data = data.mean(('lon','lat','time'))
    
    # Find the temperature of the LCL to make sure we are looking in the troposphere. Look 20 K above it.
    tlcl = data.temp.sel(pfull=data.pLCL, method='nearest').values - 20
    
    # interpolate to 800 levels for more accuracy
    p_min = 5.34e-01
    data = data.where(data.temp<tlcl,drop=True)
    data = data.interp(pfull=np.linspace(p_min,9.89e2,800))
    radcool = data.tdt_rad * 86400
    
    mask = radcool.where(radcool>threshold)
    mask = np.flip(mask[0:len(mask)-cutoff])
    temp = np.flip(data.temp[0:len(mask)-cutoff])
    pfull = np.flip(data.pfull[0:len(mask)-cutoff])

    count=0
    for i in mask.values:
        if np.isnan(i):
            i+=1
            count+=1
        else:
            break
    
    ttp = temp.values[count]
    ptp = pfull.values[count]
    ztp = 8 * np.log(1013.25/ptp)
    return ttp, ptp, ztp

def plot_temp_scm(data, threshold, width, cutoff):
    '''
    Plot the temperature (K) profile in height coordinates (km) for a single simulation.
    '''
    fig, axes = fc(1, 1, width=width, aspect=1.4, internal_pad=0)
    ax = axes[0]
    ppf.make_pretty_plot(ax, xmin=150, xmax=300, ymin=0, ymax=20, xlabel=r'Temperature / K',ylabel=r'Height / km', delxmaj=50,delxmin=10,delymaj=4,delymin=1)
    
    ttp, ptp, ztp = get_tropopause_temp(data,threshold=-0.05,cutoff=cutoff)
    temp = data.temp.mean(('lon','lat','time'))
    
    ax.plot(temp, 8*np.log(1013.25/data.pfull.values), color='k', linewidth=0.75)
    ax.scatter(ttp, ztp, color='k', s=15, marker='o', zorder=10, facecolors='silver')
    
def plot_rh_scm(data, threshold, width, cutoff):
    '''
    Plot the relative humidity (\%) profile in height coordinates (km) for a single simulation.
    '''
    fig, axes = fc(1, 1, width=width, aspect=1.4, internal_pad=0)
    ax = axes[0]
    ppf.make_pretty_plot(ax, xmin=0, xmax=100, ymin=0, ymax=20, xlabel=r'RH / \%',ylabel=r'Height / km', delxmaj=50,delxmin=10,delymaj=4,delymin=1)
    
    ttp, ptp, ztp = get_tropopause_temp(data,threshold=-0.05,cutoff=cutoff)
    rh = data.rh.mean(('lon','lat','time'))
    ax.axhline(y=ztp, color='silver', linestyle='dashed',linewidth=0.5,zorder=1)

    
    ax.plot(rh, 8*np.log(1013.25/data.pfull.values), color='k', linewidth=0.75)
    ax.scatter(ttp, ztp, color='k', s=15, marker='o', zorder=10, facecolors='silver')
    
def plot_radcool_scm(data, threshold, width, cutoff):
    '''
    Plot the radiative cooling (K/day) profile in height coordinates (km) for a single simulation.
    '''
    fig, axes = fc(1, 1, width=width, aspect=1.4, internal_pad=0)
    ax = axes[0]
    ppf.make_pretty_plot(ax, xmin=-2, xmax=2, ymin=0, ymax=20, xlabel=r'Radiative Heating / K day$^{-1}$',ylabel=r'Height / km', delxmaj=1,delxmin=0.2,delymaj=4,delymin=1)
    
    ttp, ptp, ztp = get_tropopause_temp(data,threshold=-0.05,cutoff=cutoff)
    radcool = data.tdt_rad.mean(('lon','lat','time')) * 86400
    
    ax.axvline(x=0, color='silver', alpha=0.75, linewidth=0.5)
    ax.plot(radcool, 8*np.log(1013.25/data.pfull.values), color='k', linewidth=1)
    ax.scatter(-0.05, ztp, color='k', s=15, marker='o', zorder=10, facecolors='silver')
    
def plot_lapse_scm(data, threshold, width, cutoff):
    '''
    Plot the lapse rate in height coordinates (K) for a single simulation
    '''
    fig, axes = fc(1, 1, width=width, aspect=1.4, internal_pad=0)
    ax = axes[0]
    ppf.make_pretty_plot(ax, xmin=0, xmax=10, ymin=0, ymax=20, xlabel=r'Lapse rate / K km$^{-1}$',ylabel=r'Height / km', delxmaj=2,delxmin=1,delymaj=4,delymin=1)

    ttp, ptp, ztp = get_tropopause_temp(data,threshold=-0.05,cutoff=cutoff)
    radcool = data.tdt_rad.mean(('lon','lat','time')) * 86400
    lapse = (data.pfull/data.temp * tc.g/tc.Rd * data.temp.differentiate('pfull')*1000).mean(('lon','lat','time'))
    
    lapse_trop = lapse.where(lapse.pfull<data.pLCL.mean().values,drop=True).where(lapse.pfull>ptp,drop=True)
    print(lapse_trop.mean().values)
    ax.plot(lapse, 8*np.log(1013.25/data.pfull.values), color='k', linewidth=1)
    ax.scatter(9.8, ztp, color='k', s=15, marker='o', zorder=10, facecolors='silver')

def plot_qv_scm(data, threshold, width, cutoff):
    '''
    Plot the specific humidity (g kg^-1) profile in height coordinates (km) for a single simulation.
    '''
    fig, axes = fc(1, 1, width=width, aspect=1.4, internal_pad=0)
    ax = axes[0]
    ppf.make_pretty_plot(ax, xmin=1e-7, xmax=1e-3, ymin=0, ymax=20, xscalelog=True,xlabel=r'$q_v$ / g kg$^{-1}$',ylabel=r'Height / km',delymaj=4,delymin=1)
    
    ax.set_xticks([1e-7,1e-5,1e-3,1e-1])
    
    ttp, ptp, ztp = get_tropopause_temp(data,threshold=-0.05,cutoff=cutoff)
    qv = data.sphum.mean(('lon','lat','time'))
    ax.axhline(y=ztp, color='silver', linestyle='dashed',linewidth=0.5,zorder=1)
    
    ax.plot(qv, 8*np.log(1013.25/data.pfull.values), color='k', linewidth=0.75)
    ax.scatter(ttp, ztp, color='k', s=15, marker='o', zorder=10, facecolors='silver')
    plt.xticks(rotation=45,size='small')
    
def ssm_tp(RH,kaptp,data):
    D = 1.5
    Gamma, g = (7*10**-3, 9.81)
    Rv, Rd, L = (461, 287, 2.5 * 10**6)
    Pvinf = 2.5 * 10**11
    Tstrat = 200
    Ts = data.t_surf.mean(('lon','lat','time')).values
    temp = data.temp.mean(('lon','lat','time'))
    pref = 500
    Tref = float(temp.sel(pfull=pref, method='nearest').values)
#     Tref = 260
    Tav = (Ts + Tstrat)/2
    WVP0 = Tav * RH * Pvinf / (Gamma * L)
    Tstar = L * Rd * Gamma / (g * Rv)
#     print("Ts = "+str(Ts))
#     print("Tref = "+str(Tref))

    return Tstar / lambertw( Tstar/Tref * (D*WVP0*kaptp)**(Rd*Gamma/g) )

def plot_radcool_scm_sst(data_list, threshold, width, cutoff, spectral):
    '''
    Plot the radiative cooling (K/day) profile in temperature coordinates (km) for multiple simulations.
    '''
    fig, axes = fc(1, 1, width=width, aspect=1.4, internal_pad=0)
    ax = axes[0]
    ppf.make_pretty_plot(ax, xmin=-0.1, xmax=1.1, ymin=320, ymax=170, xlabel=r'Normalized Radiative Heating / -',ylabel=r'Temperature / K', delxmaj=1,delxmin=0.2,delymaj=25,delymin=5)
    
    from matplotlib.colors import to_rgba
    
    colormap_name = 'coolwarm'
    colormap = plt.get_cmap(colormap_name)

    # Define the number of colors you want in the array
    num_colors = 7

    # Create an array of equally spaced values between 0 and 1
    values = np.linspace(0, 1, num_colors)

    # Get the RGB values for each value in the array using the chosen colormap
    colors = [to_rgba(colormap(value)) for value in values]

    count=0
    for data in data_list:
        count+=1
        ttp, ptp, ztp = get_tropopause_temp(data,threshold=-0.05,cutoff=cutoff)
        if spectral:
            data = data.mean('time')
        else:
            pass
        tlcl = data.temp.sel(pfull=data.pLCL, method='nearest').values
        data = data.where(data.temp<tlcl,drop=True)
        if spectral:
            radcool = data.tdt_rad.mean(('lon','lat')) * 86400
        else:
            radcool = data.tdt_rad.mean(('lon','lat','time')) * 86400
        ts = data.t_surf.mean().values

        ax.axvline(x=0, color='silver', alpha=0.75, linewidth=0.5)
        
        if spectral:
            ax.plot(radcool/radcool.min().values, data.temp.mean(('lon','lat')), color=colors[count], linewidth=1)
                
        else:
            ax.plot(radcool/radcool.min().values, data.temp.mean(('lon','lat','time')), color=colors[count], linewidth=1)
        ax.scatter(0, ttp, color=colors[count], s=5, marker='o', zorder=10, facecolors=colors[count])
        ax.scatter(0, ts, color=colors[count], s=5, marker='o', zorder=10, facecolors=colors[count])
        
def plot_rad_tropopause_zonal(data_list, threshold, width, cutoff, vertical_coordinate):
    # 
    fig, axes = fc(1, 1, width=width, aspect=1/1.4, internal_pad=0)
    ax0 = axes[0]
    ppf.make_pretty_plot(ax0, xmin=-90, xmax=90, ymin=210, ymax=170, xlabel=r'Latitude / deg',ylabel=r'$T_{tp}$ / K', delxmaj=45,delxmin=5,delymaj=20,delymin=2)
    ax0.set_yticks([170,190,210])
    
    fig, axes = fc(1, 1, width=width, aspect=1/1.4, internal_pad=0)
    ax1 = axes[0]
    ppf.make_pretty_plot(ax1, xmin=-90, xmax=90, ymin=10, ymax=20, xlabel=r'Latitude / deg',ylabel=r'$z_{tp}$ / km', delxmaj=45,delxmin=5,delymaj=2,delymin=1)
    
    fig, axes = fc(1, 1, width=width, aspect=1/1.4, internal_pad=0)
    ax2 = axes[0]
    ppf.make_pretty_plot(ax2, xmin=-90, xmax=90, ymin=1000, ymax=0, xlabel=r'Latitude / deg',ylabel=r'$p_{tp}$ / hPa', delxmaj=45,delxmin=5,delymaj=200,delymin=50)
    
    colors=['r','k','b']
    colors = ['#0b1d78','#006ac2','#00b5ec','#fbb862', '#ee7d4f', '#8d0006']
    
    count=0
    for data in data_list:
        ttp_list = []
        ptp_list = []
        ztp_list = []
        ts_list = []
        
        count+=1
        for lat in data.lat.values:
            ttp, ptp, ztp = get_tropopause_temp_zonal(data.sel(lat=lat),threshold=threshold,cutoff=cutoff)
            ttp_list.append(ttp)
            ptp_list.append(ptp)
            ztp_list.append(ztp)
            ts_list.append(data.sel(lat=lat).mean(('lon','time')))
            
        ax0.plot(data.lat.values, ttp_list, color=colors[count-1], linewidth=0.75, zorder=count)
        ax1.plot(data.lat.values, ztp_list, color=colors[count-1], linewidth=0.75, zorder=count)
        ax2.plot(data.lat.values, ptp_list, color=colors[count-1], linewidth=0.75, zorder=count)

def get_tropopause_temp_zonal(data, threshold,cutoff):
    '''
    Get the tropopause temperature from using the radiative cooling profile       (K/day)
    '''
    
    data = data.mean(('lon','time'))
    
    # Find the temperature of the LCL to make sure we are looking in the troposphere. Look 10 K above it.
    tlcl = data.temp.sel(pfull=data.pLCL, method='nearest').values - 10
    
    # interpolate to 800 levels for more accuracy
    p_min = 5.34e-01
    data = data.interp(pfull=np.linspace(p_min,9.89e2,800))
    data = data.where(data.temp<tlcl)
    
#     data = data.where(data.pfull>50,drop=True)
    radcool = data.tdt_rad * 86400
    
     # find where the threshold is exceeded
    mask = radcool.where(radcool>threshold)
    
     # flip the arrays so that the lowest model levels are now at the beginning of the arrays
    mask = np.flip(mask[0:len(mask)-cutoff])
    temp = np.flip(data.temp[0:len(mask)-cutoff])
    pfull = np.flip(data.pfull[0:len(mask)-cutoff])

    # keep moving up the levels until the threshold condition is satisfied
    count=0
    for i in mask.values:
        if np.isnan(i):
            i+=1
            count+=1
        else:
            break
    
    ttp = temp.values[count]
    ptp = pfull.values[count]
    ztp = 8 * np.log(1013.25/ptp)
    return ttp, ptp, ztp

def plot_lapse_tropopause_zonal(data_list, threshold, average,width,scm,vertical_coordinate):
    '''
    Plot the zonal mean tropopause temperature
    '''
    fig, axes = fc(1, 1, width=width, aspect=1/1.4, internal_pad=0)
    ax = axes[0]
    if vertical_coordinate=="pressure":
        ppf.make_pretty_plot(ax, xmin=-90, xmax=90, ymin=10, ymax=30, xlabel=r'Latitude / deg',ylabel=r'$z$ / km', delxmaj=45,delxmin=5,delymaj=5,delymin=1)
    if vertical_coordinate=="temperature":
        ppf.make_pretty_plot(ax, xmin=-90, xmax=90, ymin=210, ymax=130, xlabel=r'Latitude / deg',ylabel=r'$T$ / K', delxmaj=45,delxmin=5,delymaj=20,delymin=5)

    colors = ['#0b1d78','#006ac2','#00b5ec','#fbb862', '#ee7d4f', '#8d0006']
    count=0

    # analyze the annual- and zonal-mean data
    for data in data_list:
        ttp_list = []
        ts_list = []
        
        data = data.mean(('lon','time'))
        count+=1
        lapse = data.pfull/data.temp * tc.g/tc.Rd * data.temp.differentiate('pfull')
        lapse = lapse.interp(pfull=np.linspace(5.34e-01,9.89e2,800))
        data = data.interp(pfull=np.linspace(5.34e-01,9.89e2,800))
        for latitude in data.lat.values:
            column = data.sel(lat=latitude)
            tlcl_bin = column.temp.sel(pfull=column.pLCL, method='nearest').values - 30 # make sure to look in atmosphere
            column = column.where(column.temp<tlcl_bin)
            if vertical_coordinate=="pressure":
                ttp_bin = Ttp_lapse(column, threshold=threshold,average=average,temperature=False)
                ttp_bin = 8 * (np.log(1013.25/ttp_bin))
            if vertical_coordinate=="temperature":
                ttp_bin = Ttp_lapse(column, threshold=threshold,average=average)
            ttp_list.append(ttp_bin)
            ts_list.append(data.t_surf.values)

        ax.plot(data.lat.values, ttp_list, color=colors[count-1], linewidth=0.75, zorder=count) 
        
def Ttp_lapse(data, threshold, temperature=True, cutoff=35, average='none'):
    '''
    Determine the properties of the lapse rate tropopause.
    -------------
    data : data_set, which should include a vertical profile of radiative cooling
    threshold : the radiative tropopause is defined where the cooling exceeds this threshold (default: -0.05 K/day)
    rrtm : boolean for whether the rrtm radiation scheme is used
    gray : boolean for whether the gray radiation scheme is
    cutoff : number that roughly corresponds to the lifting condensation level. Tells the code to ignore grid points below this cutoff
    average : if 'none',   then don't do any averaging
              if 'zonal',  then average over 'lon' and 'time'
              if 'global', then average over 'lat', 'lon', and 'time'
    '''
#     if select_lat:
#         average=='zonal'
    if average=='none':
        pass
    elif average=='zonal':
        data = data.mean(('lon','time'))
    elif average=='global':
        data = data.mean(('lon','lat','time'))
    
    lapse = data.pfull/data.temp * tc.g/tc.Rd * data.temp.differentiate('pfull')*1000
    
    mask = lapse.where(lapse<threshold)
    mask = np.flip(mask[0:len(mask)-cutoff])
    temp = np.flip(data.temp.values[0:len(data.temp.values)-cutoff])
    pfull = np.flip(data.pfull.values[0:len(data.pfull.values)-cutoff])

    count=0
    for i in mask.values:
        if np.isnan(i):
            i+=1
            count+=1
        else:
            break
    if temperature==True:
        return temp[count]
    else:
        return pfull[count]
    
