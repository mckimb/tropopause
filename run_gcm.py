import os
import numpy as np
from isca import IscaCodeBase, DiagTable, Experiment, Namelist, GFDL_BASE
from isca.util import exp_progress

NCORES = 16
base_dir = os.path.dirname(os.path.realpath(__file__))
cb = IscaCodeBase.from_directory(GFDL_BASE)
cb.compile()

sst_pert = 0
exp = Experiment('EXPERIMENT_NAME', codebase=cb)
exp.clear_rundir()

#Tell model how to write diagnostics
diag = DiagTable()
diag.add_file('atmos_yearly', 360, 'days', time_units='days')
#Tell model which diagnostics to write
diag.add_field('dynamics', 'ps', time_avg=True)
diag.add_field('dynamics', 'bk')
diag.add_field('dynamics', 'pk')
diag.add_field('atmosphere', 'precipitation', time_avg=True)
diag.add_field('mixed_layer', 't_surf', time_avg=True)
diag.add_field('dynamics', 'sphum', time_avg=True)
diag.add_field('dynamics', 'ucomp', time_avg=True)
diag.add_field('dynamics', 'vcomp', time_avg=True)
diag.add_field('dynamics', 'temp', time_avg=True)
diag.add_field('dynamics', 'vor', time_avg=True)
diag.add_field('dynamics', 'div', time_avg=True)
diag.add_field('atmosphere', 'dt_ug_diffusion', time_avg=True)
diag.add_field('atmosphere', 'dt_vg_diffusion', time_avg=True)
diag.add_field('atmosphere', 'dt_tg_diffusion', time_avg=True)
diag.add_field('atmosphere', 'dt_qg_diffusion', time_avg=True)
diag.add_field('atmosphere', 'dt_qg_convection', time_avg=True)
diag.add_field('atmosphere', 'dt_tg_convection', time_avg=True)
diag.add_field('atmosphere', 'dt_tg_condensation', time_avg=True)
diag.add_field('atmosphere', 'dt_qg_condensation', time_avg=True)
diag.add_field('atmosphere', 'pLCL', time_avg=True)
diag.add_field('atmosphere', 'rh', time_avg=True)
#GRAY diagnostics
# diag.add_field('two_stream', 'tdt_rad', time_avg=True)
# diag.add_field('two_stream', 'tdt_solar', time_avg=True)
# diag.add_field('two_stream', 'olr', time_avg=True)
# diag.add_field('two_stream', 'swdn_toa', time_avg=True)
# diag.add_field('two_stream', 'swdn_sfc', time_avg=True)
# diag.add_field('two_stream', 'co2', time_avg=True)
# diag.add_field('two_stream', 'coszen', time_avg=True)
# RRTM diagnostics
diag.add_field('rrtm_radiation', 'toa_sw', time_avg=True)
diag.add_field('rrtm_radiation', 'olr', time_avg=True)
diag.add_field('rrtm_radiaiton', 'co2', time_avg=True)
diag.add_field('rrtm_radiation', 'coszen', time_avg=True)
diag.add_field('rrtm_radiation', 'tdt_rad', time_avg=True)
diag.add_field('rrtm_radiation', 'tdt_lw', time_avg=True)
diag.add_field('rrtm_radiation', 'tdt_sw', time_avg=True)
diag.add_field('rrtm_radiation', 'flux_lw', time_avg=True)
diag.add_field('rrtm_radiation', 'flux_sw', time_avg=True)

exp.diag_table = diag
# exp.inputfiles = inputfiles

#Define values for the 'core' namelist
exp.namelist = namelist = Namelist({
    'main_nml':{
     'days'   : 360,
     'hours'  : 0,
     'minutes': 0,
     'seconds': 0,
     'dt_atmos':600, # default is 1440 for lowres, 600 for highres
     'current_date' : [1,1,1,0,0,0],
     'calendar' : 'thirty_day'
    },
    'socrates_rad_nml': {
        'stellar_constant':1370.,
        'lw_spectral_filename':os.path.join(GFDL_BASE,'src/atmos_param/socrates/src/trunk/data/spectra/ga7/sp_lw_ga7'),
        'sw_spectral_filename':os.path.join(GFDL_BASE,'src/atmos_param/socrates/src/trunk/data/spectra/ga7/sp_sw_ga7'),
        'do_read_ozone': False,
        'dt_rad':3600, # default is 7200 for low res, 3600 for highres
        'store_intermediate_rad':True,
        'chunk_size': NCORES,
        'tidally_locked':False,
        'inc_co2': True,
        'co2_ppmv': 280.,
        'inc_o3': False,
        'inc_o2': False,
        'inc_n2o': False,
        'inc_ch4': False,
        'account_for_effect_of_ozone': False,
        'frierson_solar_rad': True,
        'account_for_effect_of_water': False,
    },
    'rrtm_radiation_nml': {
        'solr_cnst': 1370.,  #s set solar constant to 1360, rather than default of 1368.22
        'dt_rad': 3600, #Use long RRTM timestep
        'do_rad_time_avg':True,
        'dt_rad_avg':86400,
        'co2ppmv':280.*1.0,
        'do_read_ozone':False,
        'frierson_solar_rad':True,
        'dont_h2o':False,
    },
    'two_stream_gray_rad_nml': {
        'rad_scheme': 'frierson',            #Select radiation scheme to use, which in this case is Frierson
        'do_seasonal': False,                #do_seasonal=false uses the p2 insolation profile from Frierson 2006. do_seasonal=True uses the GFDL astronomy module to calculate seasonally-varying insolation.
        'atm_abs': 0.2,                      # default: 0.0
        'carbon_conc':280.0*1.0,
        'solar_constant':1370.0,
        'dt_rad_avg':7200
    },
    'idealized_moist_phys_nml': {
        'do_damping': True,
        'turb':True,
        'mixed_layer_bc':True,
        'do_virtual' :False,
        'do_simple': True,
        'roughness_mom':3.21e-05,
        'roughness_heat':3.21e-05,
        'roughness_moist':3.21e-05,
        'two_stream_gray': False,     #Use the grey radiation scheme
        'do_socrates_radiation': False,
        'do_rrtm_radiation': True,
        'convection_scheme': 'SIMPLE_BETTS_MILLER', #Use simple Betts miller convection
    },

    'vert_turb_driver_nml': {
        'do_mellor_yamada': False,     # default: True
        'do_diffusivity': True,        # default: False
        'do_simple': True,             # default: False
        'constant_gust': 0.0,          # default: 1.0
        'use_tau': False
    },

    'diffusivity_nml': {
        'do_entrain':False,
        'do_simple': True,
        'fixed_depth': True,
    },

    'surface_flux_nml': {
        'use_virtual_temp': False,
        'do_simple': True,
        'old_dtaudv': True
    },

    'atmosphere_nml': {
        'idealized_moist_model': True
    },

    #Use a large mixed-layer depth, and the Albedo of the CTRL case in Jucker & Gerber, 2017
    'mixed_layer_nml': {
        'tconst' : 285.,
        'prescribe_initial_dist':True,
        'evaporation':True,
        'depth': 1e10,                          #Depth of mixed layer used
        'albedo_value': 0.2,                  #Albedo value used
        'do_ape_sst': True,
        'sst_pert': sst_pert,
    },

    'qe_moist_convection_nml': {
        'rhbm':0.7,
        'Tmin':80.,
        'Tmax':350.
    },

    'lscale_cond_nml': {
        'do_simple':True,
        'do_evap':False
    },

    'sat_vapor_pres_nml': {
        'do_simple':True
    },

    'damping_driver_nml': {
        'do_rayleigh': True,
        'trayfric': -0.5,              # neg. value: time in *days*
        'sponge_pbottom':  150., #Setting the lower pressure boundary for the model sponge layer in Pa.
        'do_conserve_energy': True,
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

    'spectral_dynamics_nml': {
        'damping_order': 4,
        'water_correction_limit': 200.e2,
        'reference_sea_level_press':1.0e5,
        'num_levels':40,      #How many model pressure levels to use
        'valid_range_t':[50.,800.],
        'initial_sphum':[2e-6], # usual is 2e-6
        'vert_coord_option':'uneven_sigma',
        'surf_res':0.25, #Parameter that sets the vertical distribution of sigma levels # my usual is 0.25
        'scale_heights' : 7.0, # my usual is 7.0
        'exponent':7.0, # my usual is 7.0
        'robert_coeff':0.03, # usual is 0.03
    },

})

#Lets do a run!
if __name__=="__main__":

        # cb.compile()
        exp.set_resolution('T42')
        #Set up the experiment object, with the first argument being the experiment name.
        #This will be the name of the folder that the data will appear in.
        exp.run(1, use_restart=False, num_cores=NCORES, overwrite_data=False)
        for i in range(2,26):
            exp.run(i, num_cores=NCORES)
