import importer
from importer import import_all_modules
directory_path = 'simulation_files'
import_all_modules(directory_path, recursive=True)

class Controller:
    def __init__(self):
        """
        Initializes a new instance of the Controller class, which manages user equipment (UEs) and base stations (BSs) in a cellular network system.
        """
        self.UEs = []  # List[UE]
        self.BSs = []  # List[BS]
        
    def _lookup(self, items: List, identifier: int) -> Optional:
        """
        General-purpose method to retrieve an object from a list using its identifier.
        
        Parameters:
        - items (List): List of items to search through.
        - identifier (int): The identifier of the item to retrieve.
        
        Returns:
        - Optional: The item object if found, None otherwise.
        """
        if items:
            return next((item for item in items if item.get_id() == identifier), None)
        return None

    def bs_lookup(self, bs_id: int = None) -> Optional[BS]:
        """
        Retrieves a base station object from the list of base stations using its identifier.
        
        Parameters:
        - bs_id (int): The identifier of the base station to retrieve.
        
        Returns:
        - Optional[BS]: The base station object if found, None otherwise.
        """
        return self._lookup(self.BSs, bs_id) if bs_id is not None else None
    
    def ue_lookup(self, ue_id: int = None) -> Optional[UE]:
        """
        Retrieves a user equipment object from the list of UEs using its identifier.
        
        Parameters:
        - ue_id (int): The identifier of the user equipment to retrieve.
        
        Returns:
        - Optional[UE]: The user equipment object if found, None otherwise.
        """
        return self._lookup(self.UEs, ue_id) if ue_id is not None else None


    def bs_paging(self):
        """
        Implement paging to locate a UE within the network.
        """
        pass

    # def bs_handover(self, ue: UE):
    #     """
    #     Manages the handover of a UE from one base station to another based on signal-to-noise ratio (SINR) metrics.
    #     - Computes the SINR for all eligible BSs.
    #     - Assigns the UE to the BS with the highest SINR.
    #     - Triggers the intracell assignment process to adjust the UE's service quality or region.

    #     Parameters:
    #     - ue (UE): The user equipment undergoing handover.
    #     """
        
    #     # eligible_bs = copy.deepcopy(self.BSs) #copy of the list of BSs
    #     # current_bs = self.bs_lookup(bs_id = ue.bs_id)
    #     # # current_bs = None

    #     # # if ue.bs_id is not None:
    #     # #     current_bs = self.bs_lookup(bs_id = ue.bs_id)
    #     # #     eligible_bs.remove(current_bs)

    #     # if (current_bs is None) or (current_bs.get_distance(ue) >= BS_MAX_RANGE): #change BS
            
    #     #     if current_bs is not None:
    #     #         eligible_bs.remove(current_bs)
    #     #         current_bs.remove_ue(ue)
            
    #     #     max_sinr_ix = argmax(np.array([self.compute_sinr(bs, ue) for bs in eligible_bs])) #that would be the UL SINR, for pairing

    #     #     selected_bs = eligible_bs[max_sinr_ix]

    #     #     selected_bs.add_ue(ue) #==> trigger intracell assignment
    #     #     ue.bs_id = selected_bs.bs_id

    #     # if ue.moving:
    #     #     ue.moving = False
    #     ue_in_range = True

    #     current_bs = self.bs_lookup(ue.bs_id)

    #     eligible_bs = [bs for bs in self.BSs if bs.get_distance(ue) < BS_MAX_RANGE] 

    #     if current_bs is None or current_bs.get_distance(ue) >= BS_MAX_RANGE:

    #         if current_bs is not None:
    #             current_bs.remove_ue(ue)
            
    #         if not eligible_bs:
    #             print("User not reachable from BSs")
    #             print(f'Kill user {ue.ue_id}')
    #             ue_in_range = False# Early exit if no BS is eligible

    #         current_bs = max(eligible_bs, key=lambda bs: self.compute_sinr(bs, ue, eligible_bs))

    #         current_bs.add_ue(ue) #here will trigger the intra-cell region assignment
            
    #         ue.bs_id = current_bs.bs_id

    #     ue.moving = False

    #     self.compute_sinr(bs = current_bs, ue=ue, eligible_bs = eligible_bs) #update the sinr metrics

    #     return ue_in_range

    # def compute_sinr(self, bs: BS, ue: UE, eligible_bs: List[BS])->float:
    #     """
    #     Computes the downlink SINR for a given UE at a specified base station.
        
    #     Parameters:
    #     - bs (BS): The base station.
    #     - ue (UE): The user equipment.
        
    #     Returns:
    #     - float: The computed SINR.
    #     """

    #     dl_sinr = None 

    #     ue_just_spawned = ue.dl_sinr is None

    #     pairing_flag = ue.moving or ue_just_spawned

    #     bs_interference_set = [b for b in eligible_bs if b != bs]

    #     noise = bs.bw * bs.noise_density

    #     inter_cell_interf = sum([b.compute_link_gain(ue)*(b.p_tx_in + b.p_tx_out) for b in bs_interference_set]) #in the case some BS do not have the same power or gain

    #     interference = noise + inter_cell_interf

    #     link_gain = bs.compute_link_gain(ue)

    #     if pairing_flag or not ue.inner_region:

    #         intra_cell_interf = link_gain*bs.p_tx_in

    #         interference += intra_cell_interf 

    #         dl_sinr = (link_gain*bs.p_tx_out)/interference
    #     else:
    #         dl_sinr = (link_gain*bs.p_tx_in)/interference
        
    #     if not pairing_flag:
    #         ue.dl_sinr = dl_sinr
    #         print('UE_ID: ', ue.ue_id, ' SINR: ', ue.dl_sinr)
    #     # print(link_gain*bs.p_tx_in)
    #     # print(interference)

    #     return dl_sinr
    def bs_handover(self, ue: UE):
        """
        Manages the handover of a UE from one base station to another based on signal-to-noise ratio (SINR) metrics.
        - Computes the SINR for all eligible BSs.
        - Assigns the UE to the BS with the highest SINR.
        - Triggers the intracell assignment process to adjust the UE's service quality or region.

        Parameters:
        - ue (UE): The user equipment undergoing handover.
        """
        
        # eligible_bs = copy.deepcopy(self.BSs) #copy of the list of BSs
        # current_bs = self.bs_lookup(bs_id = ue.bs_id)
        # # current_bs = None

        # # if ue.bs_id is not None:
        # #     current_bs = self.bs_lookup(bs_id = ue.bs_id)
        # #     eligible_bs.remove(current_bs)

        # if (current_bs is None) or (current_bs.get_distance(ue) >= BS_MAX_RANGE): #change BS
            
        #     if current_bs is not None:
        #         eligible_bs.remove(current_bs)
        #         current_bs.remove_ue(ue)
            
        #     max_sinr_ix = argmax(np.array([self.compute_sinr(bs, ue) for bs in eligible_bs])) #that would be the UL SINR, for pairing

        #     selected_bs = eligible_bs[max_sinr_ix]

        #     selected_bs.add_ue(ue) #==> trigger intracell assignment
        #     ue.bs_id = selected_bs.bs_id

        # if ue.moving:
        #     ue.moving = False
        def subroutine(ue:UE, eligible_bs: List[BS]):
            bs = max(eligible_bs, key=lambda bs: self.compute_sinr(bs = bs, ue = ue, interference_set=eligible_bs, pairing= True))
            bs.add_ue(ue) #here will trigger the intra-cell region assignment
            ue.bs_id = bs.get_id()
            #ue.dl_sinr = self.compute_sinr(bs, ue, interference_set)

        ue_in_range = True

        current_bs = self.bs_lookup(ue.bs_id)

        ue_just_spawned = current_bs is None

        eligible_bs = [b for b in self.BSs if b.get_distance(ue) < BS_MAX_RANGE]

        if not eligible_bs:
                print("User not reachable from BSs")
                print(f'Kill user {ue.ue_id}')
                ue_in_range = False# Early exit if no BS is eligible

        elif ue_just_spawned:
            subroutine(ue,eligible_bs)
        
        elif ue.moving and current_bs.get_distance(ue) >= BS_MAX_RANGE: 
                
                current_bs.remove_ue(ue)
                ue.moving = False
                subroutine(ue, eligible_bs)
        #else:
            #subroutine(current_bs, ue, eligible_bs)
        #print('UE_ID: ', ue.ue_id, ' SINR: ', ue.dl_sinr)

        return ue_in_range
        

    def compute_sinr(self, bs: BS, ue: UE, pairing:bool, interference_set: List[BS] = None)->float:
        """
        Computes the downlink SINR for a given UE at a specified base station.
        
        Parameters:
        - bs (BS): The base station.
        - ue (UE): The user equipment.
        
        Returns:
        - float: The computed SINR.
        """

        dl_sinr = .0 

        if interference_set is None:
            bs_interference_set = [b for b in self.BSs if b.get_distance(ue) < BS_MAX_RANGE]
        else:
            bs_interference_set = [b for b in interference_set if b!=bs]

        noise = bs.bw * bs.noise_density

        inter_cell_interf = sum([b.compute_link_gain(ue)*(b.p_tx_in + b.p_tx_out) for b in bs_interference_set]) #in the case some BS do not have the same power or gain

        interference = noise + inter_cell_interf

        link_gain = bs.compute_link_gain(ue)

        if pairing:

            intra_cell_interf = link_gain*bs.p_tx_in

            interference += intra_cell_interf 

            dl_sinr = (link_gain*bs.p_tx_out)/interference

        elif ue.inner_region:
              
              dl_sinr = (link_gain*bs.p_tx_in)/interference

        else:
            intra_cell_interf = link_gain*bs.p_tx_in

            interference += intra_cell_interf 

            dl_sinr = (link_gain*bs.p_tx_out)/interference

        return dl_sinr
    
    def kill_ue(self, ue: UE):
        """
        Removes a UE from the system entirely, including dissociation from the BS and marking the UE as inactive.
        
        Parameters:
        - ue (UE): The user equipment to be removed.
        """
        current_bs = self.bs_lookup(bs_id = ue.bs_id)
        current_bs.remove_ue(ue)
        ue.set_inactive() #==> here it will dissociate from the bs
        self.UEs.remove(ue)

    def add_ue(self, ue:UE):
        """
        Adds a new UE to the system, sets it as active, and performs an initial base station assignment.
        
        Parameters:
        - ue (UE): The user equipment to be added.
        """
        self.UEs.append(ue)
        ue.set_active()
        return self.bs_handover(ue)

    def gather_metrics(self):
        """
        Collects and returns the SINR metrics for each UE served by its BS in the network.
        
        Returns:
        - dict: {'median_sinr': float, 'average_sinr': float} the median and average SINR across the network.
        """
        metrics_ = [self.compute_sinr(bs=self.bs_lookup(ue.bs_id), ue=ue, pairing=False) for ue in self.UEs]
        # for ue in self.UEs:
        #     bs_metrics = [ue.dl_sinr for ue in bs.served_UEs]
        #     #print(bs)
        #     #print(bs.served_UEs)
        #     #print(bs_metrics)
        #     metrics_.extend(bs_metrics)
        rounder = lambda x: round(x,5)
        return {'median_sinr': rounder(median(metrics_)), 'average_sinr': rounder(mean(metrics_))}


    def build_ues_db(self, filename: str):
        """
        Builds and exports a database of UEs with their current information to a CSV file.
        
        Parameters:
        - filename (str): The path to the CSV file where data will be stored.
        """
        data = [ue.get_info() for ue in self.UEs]
        ues_db = pd.DataFrame(data)
        ues_db.to_csv(filename, index=False)