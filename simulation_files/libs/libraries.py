# import numpy as np
# import numpy.argmax as argmax
# import pandas as pd
# import json
# import matplotlib.pyplot as plt
# import statistics
# from statistics import median as median
# from geopy.distance import distance
# Run install.py
import subprocess
subprocess.run(['python', 'simulation_files/libs/install.py'])
#subprocess.run(['python', 'install.py'])
import numpy as np
from numpy import argmax  
import pandas as pd
import json
import matplotlib.pyplot as plt
from statistics import median
from statistics import mean 
from geopy.distance import distance
import math
import folium
from geopy.geocoders import Nominatim
from functools import partial
from geopy.location import Location
from sklearn.cluster import KMeans
from numpy.linalg import norm as l2_norm
from numpy.random import uniform as U
import simpy
import logging
import random
import copy
import tqdm
from tqdm import tqdm
import imageio
import os
from typing import Optional, List, Tuple
import sys
import json
import geopandas as gpd
np.random.seed(seed=42)