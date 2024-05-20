
earth_radius_km = 6371.0 #km
ORIGIN = (48.81778, 2.23333)
DATA_SOURCE_LINK = 'https://files.data.gouv.fr/arcep_donnees/mobile/sites/2022_T3/2022_T3_sites_Metropole.csv'
NET_WIDTH = 17.014 #km
NET_HEIGHT = 9.298 #km
SITE_TYPE = 'site_4g'
OPERATOR = 'Orange'
GEO_AREA = 'Paris'
TILE_SIZE = 25/1000  #km
MU_UE, SIGMA_UE = 5, 3  #UEs'Log Gaussian Cox process(mean,std)
POPULATION = 11000
NETWORK_DENSITY = POPULATION / 158.196172  #people/km^2
OPERATOR_MARKET_SHARE = 0.385
ACTIVE_UE_FRACTION = 0.1
UE_MAX_TILE_DISTANCE = 2 #n_tiles
##Constants
W = 20 #Mhz Channel Bandwidth
N = -100 #dBm/Hz background noise
G0 = 3 #lin-scale, beamforming gain
BS_P_TX = 25 #dBm
S = 3 #lin-scale
ALPHA = 2 #lin-scale
BS_MAX_RANGE = 10 #km

def simulation_parms():
    # Get all globals defined in this module
    for name, val in globals().items():
        if not name.startswith('__') and not callable(val):
            print(f"{name}: {val}")

