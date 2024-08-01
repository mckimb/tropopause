import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import cm 
from faceted import faceted as fc
import numpy as np

# LINKS TO SHADES OF GRAY
# https://www.w3schools.com/colors/colors_shades.asp

# PREAMBLE FOR EVERY NOTEBOOK
# mpl.rcParams['xtick.color'] = '#696969' # '#808080' # '#A9A9A9' # 'k'
# mpl.rcParams['ytick.color'] = '#696969' # '#808080' #'#A9A9A9' # 'k'
# mpl.rcParams['text.usetex'] = True
# mpl.rc('font',**{'family':'serif','serif':['Palatino']})
# mpl.rcParams['figure.dpi']= 300

# COLOR OF THE FACE
# ax.set_facecolor('#fafafa')

# COLOR OF THE X- AND Y-LABELS
# plt.setp(ax.get_xticklabels(), color='#606060')
# plt.setp(ax.get_yticklabels(), color='#606060')

# COLOR OF THE SPINE
# for spine in ax.spines.values():
#         spine.set_edgecolor('#696969')

# COLOR OF THE TICK LABELS
# plt.setp(ax.get_xticklabels(), color='#606060') # 'k'
# plt.setp(ax.get_yticklabels(), color='#606060')



def make_pretty_plot(ax, xmin=-10, xmax=10, ymin=-10, ymax=10, xscalelog=False, yscalelog=False,
                    xlabel='', ylabel=''):
    ax.tick_params(axis='both', direction='in', top=True, right=True, 
                   left=True, bottom=True)
#     ax.spines['right'].set_visible(False)
#     ax.spines['top'].set_visible(False)
#     ax.yaxis.set_ticks_position('left')
#     ax.xaxis.set_ticks_position('bottom')

#     ax.set_facecolor('#fafafa') # face color
    plt.setp(ax.get_xticklabels(), color='k') # color of axis labels
    plt.setp(ax.get_yticklabels(), color='k')
    for spine in ax.spines.values():
            spine.set_edgecolor('k') # color of spine
    plt.setp(ax.get_xticklabels(), color='k') # color of tick labels
    plt.setp(ax.get_yticklabels(), color='k')
    plt.xticks(fontsize='small')
    plt.yticks(fontsize='small')
    ax.set_xlim([xmin,xmax])
    if xscalelog==True:
        ax.set_xscale('log')
    ax.set_ylim([ymin,ymax])
    if yscalelog==True:
        ax.set_yscale('log')
    ax.set_xlabel(xlabel, fontsize='small')
    ax.set_ylabel(ylabel, fontsize='small')
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(0.5)
#     ax.tick_params(width=0.6)
    ax.tick_params(axis='both', direction='in', length=1.5, top=True, right=True, left=True,
                   bottom=True, which='minor', width=0.3)
    ax.tick_params(axis='both', direction='in', length=4, top=True, right=True, left=True,
                   bottom=True, which='major', width=0.5)



    return ax

# Ukrainian Color Scheme
from matplotlib.colors import ListedColormap

N = 256
blue = np.ones((N,4))
blue[:,0] = np.linspace(0/256, 1, N)
blue[:, 1] = np.linspace(87/256, 1, N)
blue[:, 2] = np.linspace(184/256, 1, N)
Blues = ListedColormap(blue)

yellow = np.ones((N,4))
yellow[:,0] = np.linspace(254/256, 1, N)
yellow[:, 1] = np.linspace(221/256, 1, N)
yellow[:, 2] = np.linspace(0/256, 1, N)
Yellows = ListedColormap(yellow)
