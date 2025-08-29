# coding=utf-8
import json
import time
import os
import numpy as np
import numpy as np


def dict_to_obj(d):
    top = type('new', (object,), d)
    seqs = tuple, list, set, frozenset
    for i, j in d.items():
        if isinstance(j, dict):
            setattr(top, i, dict_to_obj(j))
        elif isinstance(j, seqs):
            setattr(top, i,
                    type(j)(dict_to_obj(sj) if isinstance(sj, dict) else sj for sj in j))
        else:
            setattr(top, i, j)
    return top


def load_config_from_json(json_path='config.json'):
    with open(json_path, 'r') as fr:
        data = json.load(fr)
        # print(data)
    return dict_to_obj(data)


def print_attrs(obj):
    print("****PRINT ATTRS****")
    ATTRS = ''
    for name in dir(obj):
        if not name.startswith('__'):
            ATTRS += "{}:{}\n".format(name, getattr(obj, name))

    print(ATTRS)
    return ATTRS

def file_print(info, logfilename='log.txt', savepath=None, debug=True):
    t = time.localtime(int(time.time()))
    ts = time.strftime('%m-%d %H:%M:%S ', t)
    if savepath is not None:
        if not os.path.exists(savepath):
            os.mkdir(savepath)
        fullfilename = os.path.join(savepath, logfilename)
    else:
        fullfilename = logfilename

    if debug:
        print(ts + info)

    with open(fullfilename, 'a') as f:
        f.write(ts + info + '\n')

def convert2binfromlist(bool_list):
    l = len(bool_list)
    res = 0
    for i in range(l):
        if bool_list[i] == 1:
            res += 2 ** (l - i - 1)
    return res

def calculate_max_distance(pred_values, target_values):
    """
    Calculate maximum distance between predicted and target trajectories.
    
    Args:
        pred_values: Dictionary with predicted values for lon, lat, alt
        target_values: Dictionary with target values for lon, lat, alt
        
    Returns:
        Maximum distance across all trajectories
    """
    max_distances = []
    
    for i in range(len(pred_values['lon'])):
        # Convert to ECEF coordinates for each trajectory
        pred_x, pred_y, pred_z = gc2ecef(
            pred_values['lon'][i], 
            pred_values['lat'][i], 
            pred_values['alt'][i] / 100
        )
        
        target_x, target_y, target_z = gc2ecef(
            target_values['lon'][i], 
            target_values['lat'][i], 
            target_values['alt'][i] / 100
        )
        
        # Calculate distance at each time step
        distances = np.sqrt((pred_x - target_x)**2 + (pred_y - target_y)**2 + (pred_z - target_z)**2)
        max_distances.append(np.max(distances))
    
    return np.mean(max_distances)

def frechet_distance(P, Q):
    """
    Calculate discrete Fréchet distance between two trajectories P and Q.
    
    Args:
        P: First trajectory as numpy array of shape (n, d)
        Q: Second trajectory as numpy array of shape (m, d)
        
    Returns:
        Fréchet distance
    """
    n, m = len(P), len(Q)
    
    # Compute distance matrix
    C = np.zeros((n, m))
    for i in range(n):
        for j in range(m):
            C[i, j] = np.linalg.norm(P[i] - Q[j])
    
    # Dynamic programming to find Fréchet distance
    ca = np.full((n, m), -1.0)
    
    def _c(i, j):
        if ca[i, j] > -1:
            return ca[i, j]
        elif i == 0 and j == 0:
            ca[i, j] = C[0, 0]
        elif i > 0 and j == 0:
            ca[i, j] = max(_c(i-1, 0), C[i, 0])
        elif i == 0 and j > 0:
            ca[i, j] = max(_c(0, j-1), C[0, j])
        elif i > 0 and j > 0:
            ca[i, j] = max(min(float(_c(i-1, j)), float(_c(i-1, j-1)), float(_c(i, j-1))), float(C[i, j]))
        else:
            ca[i, j] = float('inf')
        return ca[i, j]
    
    return _c(n-1, m-1)

def dtw_distance(x, y):
    """
    Calculate Dynamic Time Warping distance between two sequences.
    
    Args:
        x: First sequence as numpy array
        y: Second sequence as numpy array
        
    Returns:
        DTW distance
    """
    n, m = len(x), len(y)
    
    # Create cost matrix
    dtw_matrix = np.full((n+1, m+1), np.inf)
    dtw_matrix[0, 0] = 0
    
    for i in range(1, n+1):
        for j in range(1, m+1):
            cost = abs(x[i-1] - y[j-1])
            dtw_matrix[i, j] = cost + min(dtw_matrix[i-1, j],      # insertion
                                         dtw_matrix[i, j-1],      # deletion
                                         dtw_matrix[i-1, j-1])    # match
    
    return dtw_matrix[n, m]

def calculate_trajectory_frechet_distance(pred_values, target_values, horizon):
    """
    Calculate average Fréchet distance between predicted and target trajectories.
    
    Args:
        pred_values: Dictionary with predicted values for lon, lat, alt
        target_values: Dictionary with target values for lon, lat, alt
        horizon: Number of time steps to consider
        
    Returns:
        Average Fréchet distance
    """
    frechet_distances = []
    
    for i in range(len(pred_values['lon'])):
        # Create 3D trajectories
        pred_traj = np.column_stack([
            pred_values['lon'][i][:horizon],
            pred_values['lat'][i][:horizon],
            pred_values['alt'][i][:horizon] / 100  # Convert to km
        ])
        
        target_traj = np.column_stack([
            target_values['lon'][i][:horizon],
            target_values['lat'][i][:horizon],
            target_values['alt'][i][:horizon] / 100  # Convert to km
        ])
        
        fd = frechet_distance(pred_traj, target_traj)
        frechet_distances.append(fd)
    
    return np.mean(frechet_distances)

def calculate_trajectory_dtw_distance(pred_values, target_values, horizon):
    """
    Calculate average DTW distance for each dimension separately.
    
    Args:
        pred_values: Dictionary with predicted values for lon, lat, alt
        target_values: Dictionary with target values for lon, lat, alt
        horizon: Number of time steps to consider
        
    Returns:
        Dictionary with DTW distances for each dimension
    """
    dtw_distances = {'lon': [], 'lat': [], 'alt': []}
    
    for i in range(len(pred_values['lon'])):
        for dim in ['lon', 'lat', 'alt']:
            pred_seq = pred_values[dim][i][:horizon]
            target_seq = target_values[dim][i][:horizon]
            
            # Scale altitude to km for consistency
            if dim == 'alt':
                pred_seq = pred_seq / 100
                target_seq = target_seq / 100
            
            dtw_dist = dtw_distance(pred_seq, target_seq)
            dtw_distances[dim].append(dtw_dist)
    
    # Return average DTW distance for each dimension
    return {dim: np.mean(distances) for dim, distances in dtw_distances.items()}

def calculate_trajectory_dtw_distance_whole(pred_values, target_values):
    """
    Calculate average DTW distance for the whole 3D trajectory (lon, lat, alt) in ECEF coordinates.
    Each trajectory is treated as a sequence of 3D points in ECEF.
    Args:
        pred_values: Dictionary with predicted values for lon, lat, alt
        target_values: Dictionary with target values for lon, lat, alt
    Returns:
        Average DTW distance for the whole trajectory (in km)
    """
    dtw_distances = []
    for i in range(len(pred_values['lon'])):
        # Convert to ECEF coordinates (in km)
        pred_x, pred_y, pred_z = gc2ecef(
            pred_values['lon'][i],
            pred_values['lat'][i],
            pred_values['alt'][i] / 100  # Altitude in km
        )
        target_x, target_y, target_z = gc2ecef(
            target_values['lon'][i],
            target_values['lat'][i],
            target_values['alt'][i] / 100  # Altitude in km
        )
        pred_traj = np.column_stack([pred_x, pred_y, pred_z])
        target_traj = np.column_stack([target_x, target_y, target_z])
        n, m = len(pred_traj), len(target_traj)
        dtw_matrix = np.full((n+1, m+1), np.inf)
        dtw_matrix[0, 0] = 0
        for ii in range(1, n+1):
            for jj in range(1, m+1):
                cost = np.linalg.norm(pred_traj[ii-1] - target_traj[jj-1])
                dtw_matrix[ii, jj] = cost + min(
                    dtw_matrix[ii-1, jj],      # insertion
                    dtw_matrix[ii, jj-1],      # deletion
                    dtw_matrix[ii-1, jj-1]     # match
                )
        dtw_distances.append(dtw_matrix[n, m])
    return np.mean(dtw_distances)

def gc2ecef(lon, lat, alt):
    """
    Convert geodetic coordinates to ECEF (Earth-Centered, Earth-Fixed) coordinates.
    
    Args:
        lon: Longitude in degrees
        lat: Latitude in degrees
        alt: Altitude in km
        
    Returns:
        X, Y, Z coordinates in km
    """
    a = 6378.137  # km
    b = 6356.752
    lat = np.radians(lat)
    lon = np.radians(lon)
    e_square = 1 - (b ** 2) / (a ** 2)
    N = a / np.sqrt(1 - e_square * (np.sin(lat) ** 2))
    X = (N + alt) * np.cos(lat) * np.cos(lon)
    Y = (N + alt) * np.cos(lat) * np.sin(lon)
    Z = ((b ** 2) / (a ** 2) * N + alt) * np.sin(lat)
    return X, Y, Z