import importer
from importer import import_all_modules
directory_path = 'simulation_files'
import_all_modules(directory_path, recursive=True)

class Cluster():
    def __init__(self, id: int, group_velocity_mean: float =1., group_theta_mean: float =np.pi/2, group_alpha: float =1., group_variance: float =1):
        """
        Initialize a Cluster object.

        Parameters:
        - id (int): The ID of the cluster.
        - group_velocity_mean (float): The mean velocity of the group.
        - group_theta_mean (float): The mean angle of movement of the group.
        - group_alpha (float): The alpha parameter of the group.
        - group_variance (float): The variance parameter of the group.

        Attributes:
        - id (int): The ID of the cluster.
        - group_velocity_mean (float): The mean velocity of the group.
        - group_theta_mean (float): The mean angle of movement of the group.
        - group_alpha (float): The alpha parameter of the group.
        - group_variance (float): The variance parameter of the group.
        - paired_ues (list): The list of paired UEs.
        - destination (tuple): The current destination of the group.
        - generator (None or Generator): The moving generator associated with the group.
        - history (list): The history of destinations visited by the group.
        """
        self.id = id
        self.group_velocity_mean = group_velocity_mean
        self.group_theta_mean = group_theta_mean
        self.group_alpha = group_alpha
        self.group_variance = group_variance
        self.paired_ues = []
        self.destination = (None, None)
        self.generator = None
        self.history = [] #to record trajectory
    
    def get_id(self):
        return self.id

    def resample_params(self):
        """ 
        Resamples the movement parameters of the cluster using uniform distributions. This method adjusts the cluster's velocity, direction, alpha, and variance attributes.
        """
        max_velox = 50 * (1/3600) #km/s. we can set it to 50km/h for urban centers
        min_velox = 5 * (1/3600) #km/s. (assumin 5 km/h which is the average speed of a walk) Going at 5 km/h would take 18 seconds to cover one TILE_SIZE
        self.group_velocity_mean = U(min_velox, max_velox) #defined in km/h
        self.group_theta_mean = U(0, 2*np.pi)
        self.group_alpha = U(0.1, 1)
        self.group_variance = U(0.5, 5)

    def add_ue(self, ue: UE):
        """
        Adds a UE to the cluster's list of paired UEs.
        
        Parameters:
        - ue (UE): The UE object to be added to the cluster.
        """
        self.paired_ues.append(ue)
    
    def remove_ue(self, ue_id: int):
        """
        Removes a UE from the cluster based on its ID.
        
        Parameters:
        - ue_id (int): The ID of the UE to be removed.
        """
        #remove ue with matching UE_id
        for ue in self.paired_ues:
            if ue.ue_id == ue_id:
                self.paired_ues.remove(ue)

    def set_first_destination(self, position: np.ndarray):
        """
        Sets the initial destination for the cluster and initializes the movement generator.
        
        Parameters:
        - destination (tuple): The initial destination coordinates for the cluster.
        """

        self.generator = gauss_markov_trajectory(position = position, dimensions = (NET_WIDTH,NET_HEIGHT), group_velocity_mean=self.group_velocity_mean, 
                                             group_theta_mean=self.group_theta_mean, group_alpha=self.group_alpha, group_variance=self.group_variance)
        #or em_trajectory
        self.move_destination()

    def move_destination(self):
        """
        Updates the cluster's destination using the movement generator and records the new destination.
        """
        self.destination, _ , _ = next(self.generator)
        self.history.append(self.destination)  # Record each new destination