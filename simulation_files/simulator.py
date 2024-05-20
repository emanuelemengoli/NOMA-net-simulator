import importer
from importer import import_all_modules
directory_path = 'simulation_files'
import_all_modules(directory_path, recursive=True)

#following the process used in "Assessing the Performance of NOMA in a Multi-Cell Context: A General Evaluation Framework"
class SIMULATION():
    """
    A simulation framework for evaluating the performance of NOMA in a cell-free context, 
    based on dynamic user equipment (UE) behavior and base station (BS) interaction.
    """ 

    def __init__(self, controller: Controller, id:int=1, prob_map: List = None,tau: float = 1, lambda_: float = 2, mu: float = 4, epsilon: float = 0.4, move_interval: int = 3): #rho = lambda/mu < 1
        """
        Initializes the simulation environment, sets parameters for the simulation, and optionally, an initial probability map for UE distribution across tiles.

        Parameters:
        - controller (Controller): The control unit handling all UEs and BSs in the simulation.
        - prob_map (list, optional): A list indicating the initial probability of UE distribution across tiles.
        - tau (float): Interval for collecting SINR metrics.
        - lambda_ (float): Rate of UE arrivals.
        - mu (float): Rate of UE departures.
        - epsilon (float): Probability of exploration in UE movement.
        - move_interval (int): Interval in seconds between each movement update for UEs.

        Attributes:
        - n_tiles (int): Number of tiles across the simulated area.
        - num_tiles_x (int): Number of tiles along the width of the simulated area.
        - num_tiles_y (int): Number of tiles along the height of the simulated area.
        - env (simpy.Environment): The simulation environment from SimPy.
        - history_logger (logging.Logger): Logger for recording simulation history and metrics.
        """
        self.id = id
        self.controller = controller
        self.n_tiles = int((NET_WIDTH*NET_HEIGHT)// (TILE_SIZE**2))
        self.num_tiles_x = int(NET_WIDTH // TILE_SIZE)
        self.num_tiles_y = int(NET_HEIGHT // TILE_SIZE)
        self.env = simpy.Environment()
        self.tau = tau  # Interval for SINR metrics collection in seconds
        self.lambda_ = lambda_  # User birth rate
        self.mu = mu  # User lifetime rate parameter
        self.move_interval = move_interval  # Time interval for user movement
        self.epsilon = epsilon  # Probability of exploration in movement
        if prob_map is not None:
            self.prob_map = prob_map
        else:
            self.prob_map_random_initialization()
        self.clusters = None

        #run the simulation
        # Set up a logging system to collect time and SINR metrics
        #logging.basicConfig(level=logging.INFO)
        
        #logging.basicConfig(format="%(levelname)s | %(asctime)s | %(message)s")
        
        self.history_logger = logging.getLogger('history')
        # stdout = logging.StreamHandler(stream=sys.stdout)
        # stdout.setLevel(logging.INFO)
        # self.history_logger.addHandler(stdout)
        # self.history_logger.setLevel(logging.INFO)

        memory_handler = MemoryHandler()
        memory_handler.setLevel(logging.INFO)
        formatter = JsonFormatter()
        memory_handler.setFormatter(formatter)
        self.history_logger.addHandler(memory_handler)
        self.history_logger.setLevel(logging.INFO)
        self.history_logger.propagate = False

    def prob_map_random_initialization(self):
        """
        Initializes the probability map randomly if no predefined map is provided.
        """
        raw_probabilities = [random.random() for _ in range(self.n_tiles)]
        total = sum(raw_probabilities)
        self.prob_map = [prob / total for prob in raw_probabilities]


    def sample_position(self, tile_loc: Optional[int] = None, old_position: Optional[Tuple[float, float]] = None) -> Tuple[float, float]:
        """
        Samples a uniform random position within a given tile for placing a UE.
        """
        def random_offset() -> float:
            """Generates a random offset within the tile size."""
            return U(0, TILE_SIZE)

        if tile_loc is not None:
            row_index = tile_loc // self.num_tiles_x
            col_index = tile_loc % self.num_tiles_x

            tile_base_x = col_index * TILE_SIZE
            tile_base_y = row_index * TILE_SIZE

            new_position = (tile_base_x + random_offset(), tile_base_y + random_offset())
        elif old_position is not None:
            new_position = (old_position[0] + random_offset(), old_position[1] + random_offset())
        else:
            raise ValueError("Either tile_loc or old_position must be specified.")

        return new_position

    def sample_ue(self):
        """
        Randomly selects a UE from the controller's list of UEs.
        """
        return np.random.choice(self.controller.UEs)
        
    def sample_tile(self, tile_loc: int= None):
        """
        Samples a tile based on the probability map for new UE birth or based on a specific range for movement.
        """
        if tile_loc is None:
            ret = np.random.choice(range(self.n_tiles), p=self.prob_map)
        else: #given a tile location, sample a new tile within the reachable tiles
            low_bnd = max(0, tile_loc - UE_MAX_TILE_DISTANCE)
            up_bnd = min(tile_loc + UE_MAX_TILE_DISTANCE, self.n_tiles - 1)
            ret = np.random.randint(low_bnd,up_bnd)

        return ret

    def generate_bs(self, filepath: str):
        """
        Loads BS data from a CSV file and initializes BS objects within the controller.
        """
        bs_df = pd.read_csv(filepath)
        #self.controller.BSs = [BS(bs_id = bs_df.loc[i].id_station_anfr, position = (bs_df.loc[i].x,bs_df.loc[i].y))for i in bs_df.index] #position is in km from origin
        self.controller.BSs = [BS(bs_id=row.id_station_anfr, position=(row.x, row.y)) for _, row in bs_df.iterrows()]
        #to see if it runs
        self.controller.BSs = self.controller.BSs[:100]
        #logging.info(f'Number of Base Stations: {len(self.controller.BSs)}')
        print('num_BSs: ', len(self.controller.BSs))
    
    #the first pool of user can be sampled using a Log Gaussian Cox process
    def generate_UEs(self, max_cap_: int = None, verbose: bool = False):
        """
        Generates an initial population of UEs based on a log-Gaussian Cox process.
        """
        # Sample a Gaussian random variable for each tile
        gaussian_samples = np.random.normal(MU_UE, SIGMA_UE, self.n_tiles)

        # Calculate intensity and sample the number of UEs per tile
        intensity = np.exp(gaussian_samples)
        ue_counts = np.random.poisson(intensity)

        # Generate UEs uniformly within each tile
        users = []
        ue_id = 1
        ###
        if max_cap_ is not None and max_cap_ <= self.n_tiles:
            sampled_tiles = np.random.choice(self.n_tiles, max_cap_, replace=False)
        else:
            sampled_tiles = range(self.n_tiles)  # Use all tiles if max_cap_ is None or too large

        if verbose: print("Sampled Tiles: ",sampled_tiles)

        for tile_loc in sampled_tiles:
            if verbose: 
                print("Tile: ",tile_loc)
                print("Number of users to be generated: ", range(ue_counts[tile_loc]))
            for _ in range(ue_counts[tile_loc]):
                new_ue = UE(ue_id = ue_id, position = self.sample_position(tile_loc = tile_loc), tile = tile_loc)
                users.append(new_ue)
                ue_id += 1

        if verbose: print("Generated UEs: ", len(users))     
        # Select UEs based on Operator market share
        selected_active_users = np.random.choice(users, int(len(users) * OPERATOR_MARKET_SHARE * ACTIVE_UE_FRACTION), replace=False)
        # Set "ACTIVE_UE_FRACTION" of the selected UEs as active
        for ue in selected_active_users: 
            ue.set_active()
        self.controller.UEs =  selected_active_users
        print('num_active_users: ', len(self.controller.UEs))
    
    def generate_random_ues(self, total_population: int, verbose: bool = False):
        """
        Generates a random population of UEs distributed across tiles based on the self.prob_map.
        Args:
            total_population (int): The total number of UEs to generate.

        This function will generate `total_population` UEs, distribute them across tiles based
        on the probability map, and then place each UE at a random position within its assigned tile.
        """
        # Sample the tiles for each UE based on the probability map
        tiles = np.random.choice(range(self.n_tiles), size=total_population, p=self.prob_map)

        # Generate UEs within their respective tiles
        for tile in tiles:
            # Sample a position within the tile
            position = self.sample_position(tile_loc = tile)

            # Create a new UE with a unique ID
            ue_id = len(self.controller.UEs) + 1  # Unique ID for each UE
            new_ue = UE(ue_id=ue_id, position=position, tile=tile)
            if not self.controller.add_ue(ue = new_ue):
                self.ue_death(ue= new_ue)

        #if verbose:  # Assuming there is a verbose attribute for debug information
        print('num_active_users: ', len(self.controller.UEs))

    
    def ue_arrival(self, time: float = None): #randomly placing inside the tile (tile_loc is the number of which tile inside the grid)
        """
        Simulates the arrival of new UEs over time, integrating them into existing clusters.
        """
        if time is not None:
            yield self.env.timeout(time)

        ue_id = self.controller.UEs[-1].ue_id + 1 if self.controller.UEs else 1
        tile_loc = self.sample_tile()
        new_ue = UE(ue_id = ue_id, position = self.sample_position(tile_loc = tile_loc), tile = tile_loc)
        #new_ue.set_active()
        if not self.controller.add_ue(ue = new_ue):  
            yield self.env.process(self.ue_death(new_ue, 0))

        c_ix = np.argmin(np.array([l2_norm(np.array(new_ue.position) - np.array(c.destination)) for c in self.clusters]))
        self.clusters[c_ix].add_ue(ue = new_ue)
        new_ue.cluster = self.clusters[c_ix].get_id()
        # Add the user to the controller
        #self.controller.add_ue(ue = new_ue) ##chose bs to pair based on snr level

        # Schedule user death after a lifetime
        lifetime = random.expovariate(self.mu)
        self.env.process(self.ue_death(new_ue, lifetime))

        # Schedule the next user arrival
        inter_arrival_time = random.expovariate(self.lambda_)
        self.env.process(self.ue_arrival(time = inter_arrival_time))

    def ue_translation(self):
        """
        The clusters movement happens accordint to an epsilon greedy strategy, then UEs moves following the cluster destination. 
        The BS handover function checks whether the displacement has exceeded the maximum coverage radius of the current base station, before triggering a new BS pairing.
        """
        while True:
            for c in self.clusters:
                if random.random() < (1 - self.epsilon):
                    c.move_destination()
                    # Trigger user movement within the cluster
                    for ue in c.paired_ues:
                        ue.moving = True
                        ##depending on the mobility_model used
                        ue.position = self.sample_position(old_position= c.destination)
                        #ue.move() #if the displacement is above a given threshold, the handover will be triggered
                        if not self.controller.bs_handover(ue = ue):
                            yield self.env.process(self.ue_death(ue, 0))

            yield self.env.timeout(self.move_interval)


    def metrics_collection(self):
        """
        Periodically collects SINR metrics across all BS-UE pairs in the simulation.
        """
        while True:
            sinr_metrics = self.controller.gather_metrics()
            self.history_logger.info({
                'time': self.env.now,
                **sinr_metrics  # add also hyper-params
            })
            yield self.env.timeout(self.tau)

    def ue_death(self, ue: UE, lifetime: float = None):
        """
        Simulates the departure of a UE after its lifetime expires.
        """
        if lifetime is not None:
            yield self.env.timeout(lifetime)
        if ue.active:
            cluster = self.controller._lookup(self.clusters, ue.cluster)
            cluster.remove_ue(ue_id=ue.ue_id)
            self.controller.kill_ue(ue)
        
    # def ue_translation(self,ue: UE):
    #     #change coordinates of UE
    #     #trigger controller handover
    #     #ue can move up to MaxDistance tiles from the current tile
    #     ue.moving = True
    #     low_bnd = ue.tile-UE_MAX_TILE_DISTANCE if ue.tile>UE_MAX_TILE_DISTANCE else ue.tile-ue.tile 
    #     up_bnd = ue.tile+UE_MAX_TILE_DISTANCE
    #     new_tile_loc = np.random.randint(low_bnd,up_bnd)
    #     ue.position = self.sample_position(tile_loc = new_tile_loc)
    #     self.controller.bs_handover(ue = ue) #pair with the new bs in the new location

    def cluster_UEs(self, n_clusters):
        """
        Cluster UEs based on their spatial positions using K-means clustering.
        
        Parameters:
            users (list of UE): List of user equipment (UE) objects.
            n_clusters (int): Number of clusters to create.
        
        Returns:
            list of Cluster: List of Cluster objects with UEs grouped.
        """
        # Extract user positions from UE objects
        users = self.controller.UEs
        user_positions = np.array([ue.position for ue in users])

        # Apply K-means clustering
        kmeans = KMeans(n_clusters=n_clusters, n_init='auto').fit(user_positions)

        # Create Cluster objects based on the number of clusters
        self.clusters = [Cluster(id = i) for i in range(n_clusters)]
        #clusters = [Cluster(id = i) for i in range(len(np.unique(kmeans.labels_)))]
        
        for i,cluster in enumerate(self.clusters):
            cluster.resample_params()
            cluster.set_first_destination(position= kmeans.cluster_centers_[i])

        # Assign cluster labels to UEs and add them to their corresponding Cluster
        for ue, label in zip(users, kmeans.labels_):
            ue.cluster = label
            self.clusters[label].add_ue(ue)
    
    def write_log(self):
        # Ensure output directory exists
        os.makedirs('output', exist_ok=True)
        with open(f'output/simulation_logs_sim{self.id}.json', 'w') as log_file:
            for record in self.history_logger.handlers[0].records:
                log_file.write(record + '\n')


    def run(self, filepath: str, max_cap_: int = None, verbose: bool = False, n_clusters: int = 5, separate_plots: bool = True , gif_toggle: bool = False): #update description
        """
        Runs the entire simulation setup, managing UE births, movements, and SINR collection.
        """

        print('network_area: ', round(NET_WIDTH*NET_HEIGHT,3), 'km^2')
        print('n_tiles: ', self.n_tiles)

        #Populate the network with base stations
        self.generate_bs(filepath)

        # Intial static user population of the network
        #self.generate_UEs(max_cap_=max_cap_, verbose=verbose)
        self.generate_random_ues(total_population=max_cap_)

        # #plot the network
        # self.plot_net()

        # # Build the UEs database
        # self.controller.build_ues_db(filename= "ues_set.csv")

        #cluster the UES
        self.cluster_UEs(n_clusters)
        #self.cluster_UEs(k_means_cluster) #hparam
        self.controller.build_ues_db(filename='tmp')
        # Start the birt-death process
        self.env.process(self.ue_arrival())

        # Start the cluster movement process
        self.env.process(self.ue_translation())

        # Start the SINR collection process
        self.env.process(self.metrics_collection())

        for t in tqdm(range(SIM_TIME)):
            self.env.step()

        self.plot_net2D()
        self.plot_metrics_history(separate_plots=separate_plots)
        self.write_log()
            # if gif_toggle:
            #     map_update_ue_positions(map_obj, self.controller.UEs)  # Update UEs on the map
            #     filename = f"snapshot_{t}.html"
            #     save_map_state(map_obj, filename)

        # if gif_toggle:
        #     draw_cluster_trajectories(map_obj, clusters)  # Update trajectories on the map
        #     create_gif('path_to_snapshots_folder', 'simulation.gif', duration=1)

        # Run the SimPy event loop
        

        #to model the time 
        
        #poisson process to spwan and to kill users
        
        # Start the first user arrival
        


        #simulate timeline
        # T = 100 #s
        # ues_pool = generate_user(n_users, prob_map, grid) #==>return of class UEs 
        # generate_ues_clusters(n_clusters) #==> pair each user with a label indicating the cluster,
        # generate_cluster_params(n_clusters) #==> return a dict: key cluster_id, parameters for each cluster
        # for t in range(T):
        #     for c in clusters:
        #         for ue in c.paired_ues:
        #             ue.move()
        #         c.move_destination()

        #     if t%T_SAMPLE == 0:
        #         compute_analytics()
        #         update_log()

        # print_log()
        # plot_simulation2D()

        # if graphics:
        #     plot_simulation3D()


    
    def plot_net2D(self):
        """
        Visualizes the 2D layout of the network, including UEs, BSs, and clusters.
        """

        bss_net = self.controller.BSs
        ues_net = self.controller.UEs
        km_to_m = 1000

        fig, ax = plt.subplots()

        # Plot end-users and base stations
        ue_x_positions = [ue.position[0]*km_to_m for ue in ues_net]
        ue_y_positions = [ue.position[1]*km_to_m for ue in ues_net]
        
        ax.scatter(ue_x_positions, ue_y_positions, marker='.', c='gray', alpha=0.75, s=1, label='End-user')

        bs_x_positions = [bs.position[0]*km_to_m for bs in bss_net]
        bs_y_positions = [bs.position[1]*km_to_m for bs in bss_net]

        ax.scatter(bs_x_positions, bs_y_positions, marker='^', c='orange',s=10, label='Base Station')

        # Draw a square region of interest
        #ax.add_patch(plt.Rectangle((1500, 1500), 2000, 2000, fill=False, edgecolor='blue', linewidth=2))

        # Draw the boundary of the whole area
        #ax.add_patch(plt.Rectangle((0, 0), 5000, 5000, fill=False, edgecolor='red', linewidth=2))

        # Plot each cluster's trajectory
        for cluster in self.clusters:
            # Ensure that the cluster has a non-empty history to plot
            if hasattr(cluster, 'history') and cluster.history:
                x_positions = [pos[0] * km_to_m for pos in cluster.history]
                y_positions = [pos[1] * km_to_m for pos in cluster.history]
                ax.plot(x_positions, y_positions, marker='', linestyle='-', alpha=0.5, label=f'Cluster {cluster.id}')

        # Labeling the axes
        ax.set_xlabel('Distance (m)')
        ax.set_ylabel('Distance (m)')

        # Set the plot limits
        ax.set_xlim(0, NET_WIDTH*km_to_m)
        ax.set_ylim(0, NET_HEIGHT*km_to_m)

        # Add a legend
        ax.legend(loc = 'upper right')

        # Show the plot
        plt.show()

        #save the plot as png image in the folder output
        plt.savefig(f'output/2D_network_sim{self.id}.png')

    # def plot_metrics_history(self):
    #     """
    #     Plot the history of collected metrics over time.
    #     This function visualizes metrics such as SINR over the course of the simulation.
    #     """
    #     times, metrics = zip(*[(log['time'], log['sinr']) for log in self.history_logger.handlers[0].records])
    #     plt.figure(figsize=(10, 5))
    #     for idx, metric_snapshot in enumerate(metrics):
    #         plt.plot(times, metric_snapshot, label=f'Metric at {idx}')
    #     plt.xlabel('Simulation Time')
    #     plt.ylabel('Metrics Value')
    #     plt.title('Metrics Over Time')
    #     plt.legend()
    #     plt.show()

    # def plot_metrics_history(self, separate_plots=True):
    #     """
    #     Plot the history of collected metrics over time.
    #     This function visualizes various metrics collected during the simulation. It can generate either a single plot with all metrics or separate plots for each metric.

    #     Args:
    #     separate_plots (bool): If True, generate separate plots for each metric.
    #     """
    #     # Check if there is data to plot
    #     if not self.history_logger.handlers[0].records:
    #         print("No data to plot.")
    #         return

    #     # Gather times and all metrics from the records
    #     times = [log['time'] for log in self.history_logger.handlers[0].records]
    #     metric_keys = set(key for log in self.history_logger.handlers[0].records for key in log.keys() if key != 'time')
    #     hyper_params = [key for key in metric_keys if '_hyper_param' in key]
    #     metrics_to_plot = [key for key in metric_keys if '_hyper_param' not in key]

    #     if separate_plots:
    #         # Plot each metric on a separate figure
    #         for metric in metrics_to_plot:
    #             plt.figure(figsize=(10, 5))
    #             metric_values = [log[metric] for log in self.history_logger.handlers[0].records]
    #             plt.plot(times, metric_values, label=f'{metric.capitalize()}')
    #             hyper_param_values = ', '.join([key.split('_hyper_param')[0].capitalize() for key in hyper_params])
    #             plt.xlabel('Simulation Time')
    #             plt.ylabel('Metric Values')
    #             plt.title(f'{metric.capitalize()} Over Time')
    #             if hyper_param_values:
    #                 plt.legend([f'{metric.capitalize()} ({hyper_param_values})'])
    #             else:
    #                 plt.legend()
    #             plt.show()
    #     else:
    #         # Plot all metrics on one figure
    #         plt.figure(figsize=(10, 5))
    #         for metric in metrics_to_plot:
    #             metric_values = [log[metric] for log in self.history_logger.handlers[0].records]
    #             plt.plot(times, metric_values, label=f'{metric.capitalize()}')
    #         hyper_param_values = ', '.join([key.split('_hyper_param')[0].capitalize() for key in hyper_params])
    #         plt.xlabel('Simulation Time')
    #         plt.ylabel('Metric Values')
    #         plt.title('Metrics Over Time')
    #         if hyper_param_values:
    #             plt.legend([f'{metric.capitalize()} ({hyper_param_values})' for metric in metrics_to_plot])
    #         else:
    #             plt.legend()
    #         plt.show()

    def plot_metrics_history(self, separate_plots=True):
        handler = self.history_logger.handlers[0]
        if not hasattr(handler, 'records') or not handler.records:
            print("No data to plot.")
            return

        # Parse records assuming they are in JSON format
        records = [json.loads(record) for record in handler.records]  # Adjusting how records are parsed
        times = [log['time'] for log in records]
        metric_keys = {key for log in records for key in log if key != 'time'}

        if separate_plots:
            for metric in metric_keys:
                plt.figure(figsize=(10, 5))
                metric_values = [log[metric] for log in records if metric in log]  # Safety check for key existence
                plt.plot(times, metric_values, label=f'{metric.capitalize()}')
                plt.xlabel('Simulation Time')
                plt.ylabel('Metric Values')
                plt.title(f'{metric.capitalize()} Over Time')
                plt.legend()
                plt.show()
                plt.savefig(f'output/{metric}_sim{self.id}.png')
        else:
            plt.figure(figsize=(10, 5))
            for metric in metric_keys:
                metric_values = [log[metric] for log in records if metric in log]  # Safety check for key existence
                plt.plot(times, metric_values, label=f'{metric.capitalize()}')
            plt.xlabel('Simulation Time')
            plt.ylabel('Metric Values')
            plt.title('Metrics Over Time')
            plt.legend()
            plt.show()
            plt.savefig(f'output/metrics_sim{self.id}.png')
