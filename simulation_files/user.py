import importer
from importer import import_all_modules
directory_path = 'simulation_files'
import_all_modules(directory_path, recursive=True)

class UE:
    def __init__(self, ue_id: int, position: Tuple[float, float], tile: int):
        """ 
        Initialize a User Equipment (UE) object.

        Parameters:
        - ue_id (int): The ID of the UE.
        - position (tuple): The position of the UE.
        - tile (int): The tile number of the UE.

        Attributes:
        - ue_id (int): The ID of the UE.
        - position (tuple): The position of the UE.
        - tile (int): The tile number of the UE.
        - active (bool): Indicates if the UE is active.
        - inner_region (bool): Indicates if the UE is in the inner region.
        - bs_id (int): The ID of the base station the UE is connected to.
        - moving (bool): Indicates if the UE is moving.
        - cluster (None or Cluster): The cluster the UE belongs to.
        - generator (None or Generator): The moving generator associated with the UE.
        """
        self.ue_id = ue_id
        self.position = position
        self.tile = tile
        self.active = False 
        self.inner_region = False
        self.bs_id = None
        self.moving = False
        self.cluster = None
        self.generator = None
    
    def get_id(self):
        return self.ue_id

    def set_active(self):
        """ 
        Set the UE as active.
        """
        self.active = True

    def set_inactive(self):
        """ 
        Set the UE as inactive and clear its attributes.
        """
        self.active = False
        # Clear the base station
        # Sanity check
        if self.bs_id is not None: self.bs_id = None
        # Clear variables
        self.inner_region = False
        self.moving = False
        self.generator = None
        self.position = None
        self.tile = None
        self.cluster = None

    def get_info(self):
        """
        Get information about the UE.

        Returns:
        - dict: A dictionary containing the UE's information.
        """
        return {
            'UE ID': self.ue_id,
            'Position': self.position,
            'Tile': self.tile,
            'Active': self.active,
            'Inner Region': self.inner_region, 
            'BS ID': self.bs_id,
            'Moving': self.moving,
            'Generator': self.generator,
            'Cluster': self.cluster  
        }

    def get_geo_coordinate(self):
        """
        Get the geographical coordinates of the UE based on the reference position of origin.

        Returns:
        - tuple: The latitude and longitude of the UE's position.
        """
        # NB
        # source: http://www.edwilliams.org/avform147.htm
        # and https://gis.stackexchange.com/questions/2951/algorithm-for-offsetting-a-latitude-longitude-by-some-amount-of-meters/2980#2980?newreg=0bedc72752ea4e629440a761d6f4a231
        # ORIGIN = (Latitude, Longitude) 
        # self.position = (x, y) in km x=>longitude, y=>latitude

        # Convert latitude and longitude to radians
        lat_rad = math.radians(ORIGIN[0])
        # lon_rad = math.radians(ORIGIN[1])

        # Calculate the change in latitude
        delta_lat = self.position[1] / earth_radius_km

        # Calculate the change in longitude
        delta_lon = self.position[0] / (earth_radius_km * math.cos(lat_rad))

        # Convert the changes to degrees
        delta_lat_deg = math.degrees(delta_lat)
        delta_lon_deg = math.degrees(delta_lon)

        # Calculate the new latitude and longitude
        new_latitude = ORIGIN[0] + delta_lat_deg
        new_longitude = ORIGIN[1] + delta_lon_deg

        return new_latitude, new_longitude
    
    def move(self):
        """
        Move the UE based on the generator associated with the UE's cluster.
        """
        # if not self.generator:
        #     self.generator = gauss_markov_trajectory(position=self.cluster.destination, dimensions=(NET_WIDTH, NET_HEIGHT), group_velocity_mean=self.cluster.group_velocity_mean, 
        #                                      group_theta_mean=self.cluster.group_theta_mean, group_alpha=self.cluster.group_alpha, group_variance=self.cluster.group_variance)
              
        # self.position, _, _ = next(self.generator)
        pass
        #or em_trajectory