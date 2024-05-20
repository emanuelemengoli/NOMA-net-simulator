import numpy as np

def gauss_markov_trajectory(position, dimensions, velocity = None, theta = None, group_velocity_mean=1., group_theta_mean=np.pi/2, group_alpha=1., group_variance=1.):
    """
    Gauss-Markov Mobility Model, as proposed in 
    Camp, T., Boleng, J. & Davies, V. A survey of mobility models for ad hoc network research. 
    Wireless Communications and Mobile Computing 2, 483-502 (2002).

    Code inspiration: https://github.com/panisson/pymobility/blob/master/src/pymobility/models/mobility.py

    Parameters:
    - position: Tuple[float, float], the initial (x, y) position of the entity.
    - velocity: float, the initial velocity of the entity.
    - theta: float, the initial angle (radians) of movement.
    - dimensions: Tuple[int, int], the (width, height) of the simulation area.
    - group_velocity_mean: float, mean velocity of the group.
    - group_theta_mean: float, mean angle of the group.
    - group_alpha: float, tuning parameter used to vary the randomness.
    - group_variance: float, variance of randomness.

    Yields:
    - Tuple[Tuple[float, float], float, float]: The next position (x, y), updated velocity, and angle (theta).
    """
    x, y = position
    max_x, max_y = dimensions

    if velocity == None: velocity= group_velocity_mean
    
    if theta == None: theta = group_theta_mean

    g_alpha2 = 1.0 - group_alpha
    g_alpha3 = np.sqrt(1.0 - group_alpha**2) * group_variance

    while True:
        # Calculate the next position
        x += velocity * np.cos(theta)
        y += velocity * np.sin(theta)

        # Bounce off the edges
        if x < 0:
            x = -x
            theta = np.pi - theta
            group_theta_mean = np.pi-group_theta_mean #check if makes to make all the cluster change direction
        elif x > max_x:
            x = 2 * max_x - x
            theta = np.pi - theta
            group_theta_mean = np.pi-group_theta_mean

        if y < 0:
            y = -y
            theta = -theta
            group_theta_mean = -group_theta_mean
        elif y > max_y:
            y = 2 * max_y - y
            theta = -theta
            group_theta_mean = -group_theta_mean

        # Update velocity and angle based on the Gauss-Markov process
        velocity = (group_alpha * velocity +
                    g_alpha2 * group_velocity_mean +
                    g_alpha3 * np.random.normal(0.0, 1.0))

        theta = (group_alpha * theta +
                 g_alpha2 * group_theta_mean +
                 g_alpha3 * np.random.normal(0.0, 1.0))

        # Yield the next state
        yield (x, y), velocity, theta
