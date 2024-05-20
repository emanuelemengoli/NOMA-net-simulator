import importer
from importer import import_all_modules
directory_path = 'simulation_files'
import_all_modules(directory_path, recursive=True)


class BS():
    def __init__(self, bs_id: int, position = Tuple[float, float]):

        """
        Initializes a new instance of the BS (Base Station) class which represents a base station in a cellular network.

        Parameters:
        - bs_id (int): The identifier for the base station.
        - position (tuple): The geographic coordinates of the base station.
        
        Attributes:
        - bw (float): Bandwidth available to the base station.
        - g0 (float): Reference path loss at a distance of 1 meter.
        - p_tx (float): Transmit power of the base station.
        - noise_density (float): Noise density in the network.
        - shadowing (float): Shadowing effect on the signal, specific to each UE.
        - alpha (float): Path loss exponent.
        """

        self.bs_id = bs_id
        self.position = position
        self.bw= W 
        self.g0 = G0
        self.p_tx = BS_P_TX 
        self.noise_density = N
        self.shadowing = S
        self.alpha = ALPHA
        self.p_tx_in, self.p_tx_out = None, None
        self.served_UEs = [] #memorize the UE object
        self.update_snr_thr = True
        self.snr_thr = None
        self.power_adjustment()

    def get_id(self):
        return self.bs_id

    def power_adjustment(self):
        """
        Adjusts the power output levels for the base station.
        """
        #variables: self.p_tx_in, self.p_tx_out
        #max_cap: BS_P_TX
        #contraints: self.p_tx_out > self.p_tx_in as long as h_out < h_in
        #obj function to max
        
        self.p_tx_out = 0.6* BS_P_TX
        self.p_tx_in = BS_P_TX - self.p_tx_out
    
    #need to implement the alpha-fairness formulation
    def retreive_snr_thr(self): #as I have not implmented the BO obj function yet, I use a median to compute the snr threshold
        """
        Computes the SNR threshold by taking the median of the SNRs of all served UEs. This is used to determine which UEs are in the inner region of coverage.
        """
        assert self.served_UEs, 'Empty user set'
        snr_lst = [self.compute_snr(ue) for ue in self.served_UEs]
        self.snr_thr = median(snr_lst)
    
    def add_ue(self, ue: UE): 
        """
        Adds a UE to the list of UEs served by this base station. Also triggers intra-cell assignment to determine if the UE is within the inner region based on SNR.
        
        Parameters:
        - ue (UE): The user equipment instance to be added.
        """
        self.served_UEs.append(ue)
        self.intra_cell_assignment(ue) 
        self.update_snr_thr = True

    def intra_cell_assignment(self, ue: UE):
        """
        Determines whether the given UE is within the inner region of coverage based on its SNR compared to the SNR threshold.
        
        Parameters:
        - ue (UE): The user equipment instance being evaluated.
        """
        if (self.snr_thr is None) or self.update_snr_thr:
            self.retreive_snr_thr()
            self.update_snr_thr = False
        #agin consider that the real SNR measurement is backpropagated from the UE to the BS, 
        #here we compute at the BS side as it was the DL SNR
        snr_ue = self.compute_snr(ue)
        ue.inner_region = snr_ue > self.snr_thr 
        # if snr_ue > self.snr_thr:
        #     inner_flag = True
        # else:
        #     inner_flag = False
        # ue.inner_region = inner_flag

    def remove_ue(self, ue: UE):
        """
        Removes a UE from the base station's list of served UEs and recalculates the SNR threshold.
        
        Parameters:
        - ue (UE): The user equipment instance to be removed.
        """
        self.served_UEs.remove(ue)
        if self.served_UEs: 
            self.retreive_snr_thr() #update the snr_thr
    
    def get_distance(self, ue:UE):
        """
        Calculates the eucledian distance between the base station and a UE.
        
        Parameters:
        - ue (UE): The user equipment whose distance is to be calculated.
        """
        return math.sqrt(sum((x1 - x2) ** 2 for x1, x2 in zip(self.position, ue.position)))

    def compute_link_gain(self, ue:UE):
        """
        Computes the link gain between the base station and a UE using a standard path loss model.
        
        Parameters:
        - ue (UE): The user equipment for which link gain is computed.
        """
        d = self.get_distance(ue)
        return self.g0 * self.shadowing * (d**(-self.alpha)) #standard path loss model in nominal, LOS conditions

    def compute_snr(self, ue: UE):
        """
        Calculates the Signal-to-Noise Ratio (SNR) for a UE based on its position, either the inner or outer region.
        
        Parameters:
        - ue (UE): The user equipment for which SNR is computed.
        """

        # if ue.inner_region:
        #     pw = self.p_tx_in
        # else:
        #     pw = self.p_tx_out
        # snr = (pw*self.compute_link_gain(ue=ue))/(self.noise_density*self.bw)
        # return snr
        pw = self.p_tx_in if ue.inner_region else self.p_tx_out
        return (pw * self.compute_link_gain(ue)) / (self.noise_density * self.bw)
  
    #eventual functions that allow to modify simulation parameters during the simulation
    def modify_params(self, bw: float = None, g0: float = None,shadowing: float = None,alpha: float =None, p_tx: float = None, noise: float = None):
        """
        Modifies various parameters of the base station dynamically, based on the inputs provided. Each parameter is optional.
        
        Parameters:
        - bw (float): New bandwidth.
        - g0 (float): New reference path loss value.
        - p_tx (float): New transmit power.
        - noise (float): New noise density.
        - shadowing (float): New shadowing effect.
        - alpha (float): New path loss exponent.
        """
        if bw is not None:
            self.bw = bw
        if g0 is not None:
            self.g0 = g0
        if p_tx is not None:
            self.p_tx = p_tx
        if noise is not None:
            self.noise_density = noise
        if shadowing is not None:
            self.shadowing = shadowing
        if alpha is not None:
            self.alpha = alpha