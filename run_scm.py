import os, sys
import numpy as np
from isca import ColumnCodeBase, DiagTable, Experiment, Namelist, GFDL_BASE

### to create ozone file: #o3
sys.path.append("/home/links/bam218/Tropopause/")
from scm_interp_routine import scm_interp, global_average_lat_lon # o3

# column model only uses 1 core
NCORES = 1

# compile code
base_dir = os.path.dirname(os.path.realpath(__file__))
cb = ColumnCodeBase.from_directory(GFDL_BASE)
cb.compile()

# common parameter settings that are varied in experiments
insol_multiply = 1 # multiplicated factor to vary the solar insolation
b_multiply = 1. # muliplicative factor to vary the strength of the greenhouse effect in gray radiation schemes
co2_multiply = 1. # PI = 280 # multiplicative factor to vary the amount of CO2 present in the spectral radiation scheme
RH = 0.7 # column relative humidity
sst = 290 # sea surface temperature
exp = Experiment('Experiment_Name', codebase=cb)

exp.clear_rundir()

#Tell model how to write diagnostics
diag = DiagTable()
diag.add_file('atmos_yearly', 360, 'days', time_units='days')

#Tell model which diagnostics to write
diag.add_field('column', 'ps', time_avg=True)
diag.add_field('column', 'bk')
diag.add_field('column', 'pk')
diag.add_field('atmosphere', 'precipitation', time_avg=True)
diag.add_field('mixed_layer', 't_surf', time_avg=True)
diag.add_field('mixed_layer', 'flux_lhe', time_avg=True)
diag.add_field('mixed_layer', 'flux_t', time_avg=True)
diag.add_field('column', 'sphum', time_avg=True)
diag.add_field('column', 'ucomp', time_avg=True)
diag.add_field('column', 'vcomp', time_avg=True)
diag.add_field('column', 'temp', time_avg=True)
diag.add_field('column', 'height', time_avg=True)

# RRTM
diag.add_field('rrtm_radiation', 'toa_sw', time_avg=True)
diag.add_field('rrtm_radiation', 'olr', time_avg=True)
diag.add_field('rrtm_radiation', 'co2', time_avg=True)
# diag.add_field('rrtm_radiation', 'ozone', time_avg=True) # o3
diag.add_field('rrtm_radiation', 'coszen', time_avg=True)
diag.add_field('rrtm_radiation', 'tdt_rad', time_avg=True)
diag.add_field('rrtm_radiation', 'tdt_lw', time_avg=True)
diag.add_field('rrtm_radiation', 'tdt_sw', time_avg=True)
diag.add_field('rrtm_radiation', 'flux_lw', time_avg=True)
diag.add_field('rrtm_radiation', 'flux_sw', time_avg=True)
# GRAY
# diag.add_field('two_stream', 'tdt_rad', time_avg=True)
# diag.add_field('two_stream', 'tdt_solar', time_avg=True)
# diag.add_field('two_stream', 'olr', time_avg=True)
# diag.add_field('two_stream', 'swdn_toa', time_avg=True)
# diag.add_field('two_stream', 'swdn_sfc', time_avg=True)
# diag.add_field('two_stream', 'co2', time_avg=True)
# diag.add_field('two_stream', 'coszen', time_avg=True)
# other variables
diag.add_field('atmosphere', 'dt_ug_diffusion', time_avg=True)
diag.add_field('atmosphere', 'dt_vg_diffusion', time_avg=True)
diag.add_field('atmosphere', 'dt_tg_diffusion', time_avg=True)
diag.add_field('atmosphere', 'dt_qg_diffusion', time_avg=True)
diag.add_field('atmosphere', 'dt_qg_convection', time_avg=True)
diag.add_field('atmosphere', 'dt_tg_convection', time_avg=True)
diag.add_field('atmosphere', 'dt_tg_condensation', time_avg=True)
diag.add_field('atmosphere', 'dt_qg_condensation', time_avg=True)
diag.add_field('atmosphere', 'rh', time_avg=True)
diag.add_field('atmosphere', 'convection_rain', time_avg=True)
diag.add_field('atmosphere', 'cape', time_avg=True)
diag.add_field('atmosphere', 'cin', time_avg=True)
diag.add_field('atmosphere', 'pLCL', time_avg=True)
diag.add_field('atmosphere', 'pLZB', time_avg=True)
diag.add_field('atmosphere', 'pshallow', time_avg=True)
diag.add_field('column', 'dt_a', time_avg=True)
# the following diagnostics don't work
# diag.add_field('atmosphere', 'dt_qg_total', time_avg=True)
# diag.add_field('atmosphere', 'pbl_height', time_avg=True)
# diag.add_field('atmosphere', 'kLZB', time_avg=True)
# diag.add_field('atmosphere', 'z_pbl', time_avg=True)

exp.diag_table = diag

#Define values for the 'core' namelist
exp.namelist = namelist = Namelist({
    'main_nml':{
     'days'   : 360,
     'hours'  : 0,
     'minutes': 0,
     'seconds': 0,
     'dt_atmos':1440,
     'current_date' : [1,1,1,0,0,0],
     'calendar' : 'thirty_day'
         },

    'atmosphere_nml': {
        'idealized_moist_model': True
    },

    'column_nml': {
        'lon_max': 1, # number of columns in longitude, default begins at lon=0.0
        'lat_max': 1, # number of columns in latitude, precise
                      # latitude can be set in column_grid_nml if only 1 lat used.
        'num_levels': 80,  # number of levels
        'initial_sphum': 1e-3, # default 1e-6
        'vert_coord_option': 'uneven_sigma',
        'surf_res':0.25,
        'scale_heights':7.0,
        'exponent':7.0,
        'robert_coeff':0. # 0 for column model
    },

    'column_grid_nml': {
        'lat_value': np.rad2deg(np.arcsin(1/np.sqrt(3))) # set latitude to that which causes insolation in frierson p2 radiation to be insolation / 4.
    },

    # set initial condition, NOTE: currently there is not an option to read in initial condition from a file.
    'column_init_cond_nml': {
        'initial_temperature': sst-1, # initial atmospheric temperature 264, the sst in isca is the initial_temperature +1.
        'surf_geopotential': 0.0, # applied to all columns
        'surface_wind': 5. # as described above
    },

    'idealized_moist_phys_nml': {
        'do_damping': False, # no damping in column model, surface wind prescribed
        'turb':True,        # DONT WANT TO USE THIS, BUT NOT DOING SO IS STOPPING MIXED LAYER FROM WORKING
        'mixed_layer_bc':True, # need surface, how is this trying to modify the wind field? ****
        'do_simple': True, # simple RH calculation
        'roughness_mom': 3.21e-05, # DONT WANT TO USE THIS, BUT NOT DOING SO IS STOPPING MIXED LAYER FROM WORKING
        'roughness_heat':3.21e-05,
        'roughness_moist':3.21e-05,
        'two_stream_gray': False,     #Use grey radiation
        'do_rrtm_radiation': True,
        'convection_scheme': 'SIMPLE_BETTS_MILLER', #Use the simple Betts Miller convection scheme
        'do_cloud_simple': False,
    },
    'rrtm_radiation_nml': {
        'solr_cnst': 1370.*insol_multiply,  #s set solar constant to 1360, rather than default of 1368.22
        'dt_rad': 7200, #Use long RRTM timestep
        'do_rad_time_avg':True,
        'dt_rad_avg':86400,
        'co2ppmv':280.*co2_multiply,
        # 'do_read_ozone':True, # o3
        'frierson_solar_rad': True,
        'dont_h2o': False,

    },
    'two_stream_gray_rad_nml': {
        # 'rad_scheme': 'frierson',            #Select radiation scheme to use, which in this case is Frierson
        'rad_scheme': 'BYRNE',            #Select radiation scheme to use, which in this case is Frierson
        'do_seasonal': False,                #do_seasonal=false uses the p2 insolation profile from Frierson 2006. do_seasonal=True uses the GFDL astronomy module to calculate seasonally-varying insolation.
        'atm_abs': 0.,                      # default: 0.0
        'carbon_conc':280.0*1,
        'odp':1.0*co2_multiply,
                             # default:
        # 'solar_constant': 1370.
        # 'bog_b': 1997.9, # absorption coefficient in Byrne longwave
        'bog_b': 100*b_multiply, # absorption coefficient in Byrne longwave
        'bog_a': 0,
        'bog_mu': 0,
        'dt_rad_avg':7200,
        'solar_constant': 1370*insol_multiply,
    },
    'qe_moist_convection_nml': {
        'rhbm':RH, # rh criterion for convection
        'Tmin':80, # min temperature for convection scheme look up tables
        'Tmax':350.  # max temperature for convection scheme look up tables
    },

    'lscale_cond_nml': {
        'do_simple':True, # only rain
        'do_evap':False,  # no re-evaporation of falling precipitation
    },

    'surface_flux_nml': {
        'use_virtual_temp': False, # use virtual temperature for BL stability
        'do_simple': True,
        'old_dtaudv': True
    },

    'vert_turb_driver_nml': { # DONT WANT TO USE THIS, BUT NOT DOING SO IS STOPPING MIXED LAYER FROM WORKING
        'do_mellor_yamada': False,     # default: True
        'do_diffusivity': True,        # default: False
        'do_simple': True,             # default: False
        'constant_gust': 0.0,          # default: 1.0
        'use_tau': False
    },

    'diffusivity_nml': {
        'do_entrain': False,
        'do_simple': True,
        'fixed_depth': True, # default: False
    },

    #Use a large mixed-layer depth, and the Albedo of the CTRL case in Jucker & Gerber, 2017
    'mixed_layer_nml': {
        'tconst' : sst, # default: 285
        'prescribe_initial_dist':False,
        'evaporation':True,
        'depth': 1e10,                          #Depth of mixed layer used, 20 mostly
        # 'depth': 2.5,
        'albedo_value': 0.20,                  #Albedo value used
    },

    'sat_vapor_pres_nml': {
        'do_simple':True,
    },

    # FMS Framework configuration
    'diag_manager_nml': {
        'mix_snapshot_average_fields': False  # time avg fields are labelled with time in middle of window
    },
    'fms_nml': {
        'domains_stack_size': 600000                        # default: 0
    },
    'fms_io_nml': {
        'threading_write': 'single',                         # default: multi
        'fileset_write': 'single',                           # default: multi
    },
})

#Lets do a run!
if __name__=="__main__":
    # o3
    # ds = scm_interp(filename=os.path.join(GFDL_BASE,'input/rrtm_input_files/ozone_1990.nc'),
               # varname='ozone_1990',
               # nlevels=80)
    # global_average_lat_lon(ds, 'ozone_1990_interp')
    # exp.namelist['rrtm_radiation_nml']['do_scm_ozone'] = True
    # exp.namelist['rrtm_radiation_nml']['scm_ozone'] = np.squeeze(ds.ozone_1990_interp_area_av.mean('time').values).tolist()
    # # o3

    # exp.run(1, use_restart=False, num_cores=NCORES)
    exp.run(1, use_restart=False, num_cores=NCORES, mpirun_opts='--bind-to-socket')
    for i in range(2,11):
    # for i in range(2,26):
        exp.run(i, num_cores=NCORES)
