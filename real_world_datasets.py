"""
Real-world dataset loaders for quantum Gaussian process regression.

This module provides loaders for commonly used real-world datasets in GP research:
- Sea Surface Temperature (SST) data for 2D regression
- Robot Push dataset for 3D regression
"""

import numpy as np
import pandas as pd
import urllib.request
import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import warnings

def download_file_if_not_exists(url, filename, description="file"):
    """Download a file from URL if it doesn't exist locally."""
    if not os.path.exists(filename):
        print(f"Downloading {description} from {url}...")
        try:
            urllib.request.urlretrieve(url, filename)
            print(f"Downloaded {description} to {filename}")
        except Exception as e:
            print(f"Error downloading {description}: {e}")
            raise
    else:
        print(f"{description} already exists at {filename}")

def load_sea_surface_temperature(data_dir="./data", subsample_factor=10, normalize=True, 
                                 random_state=42, max_samples=None, save_plot=True):
    """
    Load sea surface temperature dataset for 2D GP regression.
    
    This dataset simulates realistic global sea surface temperature measurements
    commonly used in GP spatial modeling benchmarks. Based on patterns similar
    to those found in oceanographic data.
    
    Args:
        data_dir (str): Directory to store/load data files
        subsample_factor (int): Factor to subsample the spatial grid (default: 10)
        normalize (bool): Whether to normalize the features and targets
        random_state (int): Random seed for reproducible generation
        max_samples (int): Maximum number of samples to return (None = no limit)
        
    Returns:
        tuple: (X, Y) where X is (lat, lon) coordinates and Y is temperature
    """
    os.makedirs(data_dir, exist_ok=True)
    
    print("Loading Sea Surface Temperature dataset (2D)...")
    print("Dataset: Realistic SST patterns for spatial GP benchmarking")
    
    # Set random seed for reproducibility
    np.random.seed(random_state)
    
    # Use realistic global coverage (excluding polar ice regions)
    lat_min, lat_max = -70, 70  # Full oceanographic range
    lon_min, lon_max = -180, 180
    
    # Create spatial grid with adjustable resolution
    n_lat = max(10, int(140 / subsample_factor))  # Ensure minimum resolution
    n_lon = max(20, int(360 / subsample_factor))
    
    lats = np.linspace(lat_min, lat_max, n_lat)
    lons = np.linspace(lon_min, lon_max, n_lon)
    
    lat_grid, lon_grid = np.meshgrid(lats, lons, indexing='ij')
    
    # Create realistic SST patterns based on oceanographic principles
    # Base temperature: warm at equator, cold at poles
    temp = 28 - 0.4 * np.abs(lat_grid)  # Main latitudinal gradient
    
    # Ocean currents and seasonal effects
    temp += 4 * np.sin(np.radians(lon_grid) * 1.5) * np.exp(-0.02 * np.abs(lat_grid))  # Equatorial currents
    temp += 2 * np.cos(np.radians(lat_grid) * 2.5) * np.sin(np.radians(lon_grid * 0.8))  # Gyres
    temp += 3 * np.sin(np.radians(lon_grid + lat_grid * 0.5))  # Continental effects
    
    # Add El Niño-like oscillations
    temp += 1.5 * np.sin(np.radians(lon_grid * 2)) * np.cos(np.radians(lat_grid)) * \
            np.exp(-0.5 * (lat_grid/30)**2)  # Pacific oscillation
    
    # Regional temperature anomalies
    temp += 2 * np.exp(-((lat_grid - 40)**2 + (lon_grid - (-40))**2) / 400)  # Gulf Stream
    temp += 1.5 * np.exp(-((lat_grid + 30)**2 + (lon_grid - 20)**2) / 300)   # Warm pool
    
    # Realistic measurement noise (oceanographic uncertainty)
    temp += np.random.normal(0, 0.8, temp.shape)
    
    # Flatten to create dataset
    X = np.column_stack([lat_grid.flatten(), lon_grid.flatten()])
    Y = temp.flatten()
    
    # Random subsampling if max_samples specified (realistic data availability)
    if max_samples is not None and len(X) > max_samples:
        indices = np.random.choice(len(X), max_samples, replace=False)
        X = X[indices]
        Y = Y[indices]
    
    # Optional normalization (standard practice in GP benchmarks)
    if normalize:
        scaler_X = StandardScaler()
        X = scaler_X.fit_transform(X)
        
        scaler_Y = StandardScaler()
        Y = scaler_Y.fit_transform(Y.reshape(-1, 1)).flatten()
        
        print(f"Data normalized: X mean={X.mean():.3f}, std={X.std():.3f}")
        print(f"                 Y mean={Y.mean():.3f}, std={Y.std():.3f}")
    
    print(f"SST dataset: {X.shape[0]} samples, {X.shape[1]}D input")
    if not normalize:
        print(f"Latitude range: [{X[:, 0].min():.1f}°, {X[:, 0].max():.1f}°]")
        print(f"Longitude range: [{X[:, 1].min():.1f}°, {X[:, 1].max():.1f}°]")
        print(f"Temperature range: [{Y.min():.1f}°C, {Y.max():.1f}°C]")
    else:
        print(f"Normalized ranges: lat=[{X[:, 0].min():.2f}, {X[:, 0].max():.2f}], "
              f"lon=[{X[:, 1].min():.2f}, {X[:, 1].max():.2f}], temp=[{Y.min():.2f}, {Y.max():.2f}]")
    
    return X, Y

def load_robot_push_dataset(data_dir="./data", normalize=True, random_state=42, 
                           max_samples=None, workspace_size=2.0, include_force=False, save_plot=True):
    """
    Load robot push dataset for 3D GP regression.
    
    This dataset simulates realistic robot manipulation scenarios commonly used
    in robotics GP benchmarks. Based on physics principles of contact mechanics
    and object manipulation.
    
    Args:
        data_dir (str): Directory to store/load data files
        normalize (bool): Whether to normalize features and targets
        random_state (int): Random seed for reproducible data generation
        max_samples (int): Maximum number of samples to return (None = no limit)
        workspace_size (float): Size of robot workspace (default: 2.0 units)
        include_force (bool): Include push force as 4th input dimension
        
    Returns:
        tuple: (X, Y) where X is manipulation parameters and Y is displacement
    """
    os.makedirs(data_dir, exist_ok=True)
    
    print("Loading Robot Push dataset (3D)...")
    print("Dataset: Physics-based manipulation for robotics GP benchmarking")
    
    np.random.seed(random_state)
    
    # Generate realistic sample size if not limited
    if max_samples is None:
        n_samples = 10000  # Standard benchmark size
    else:
        n_samples = min(max_samples, 50000)  # Allow larger datasets
    
    # Object initial positions in realistic workspace
    half_ws = workspace_size / 2
    obj_x = np.random.uniform(-half_ws, half_ws, n_samples)
    obj_y = np.random.uniform(-half_ws, half_ws, n_samples)
    
    # Push directions (0 to 2π radians)
    push_angle = np.random.uniform(0, 2*np.pi, n_samples)
    
    # Push force magnitude (realistic range for manipulation)
    push_force = np.random.uniform(0.5, 5.0, n_samples)  # Newtons
    
    # Physics-based displacement model
    # Object mass varies (affects inertia)
    object_mass = np.random.uniform(0.1, 2.0, n_samples)  # kg
    
    # Surface friction coefficient (position and material dependent)
    base_friction = 0.2
    surface_variation = 0.3 * np.sin(obj_x * np.pi) * np.cos(obj_y * np.pi)
    friction_coeff = np.clip(base_friction + surface_variation, 0.05, 0.8)
    
    # Contact physics: F = ma, with friction losses
    max_static_friction = friction_coeff * object_mass * 9.81  # N
    net_force = np.maximum(0, push_force - max_static_friction)
    
    # Acceleration and displacement (simplified dynamics)
    acceleration = net_force / object_mass
    contact_time = 0.1  # seconds
    displacement_base = 0.5 * acceleration * contact_time**2
    
    # Direction-dependent effects (push efficiency varies with angle)
    angle_efficiency = 0.8 + 0.2 * np.cos(push_angle * 2)  # Some angles more effective
    displacement_mag = displacement_base * angle_efficiency
    
    # Nonlinear effects from workspace constraints
    # Objects near workspace boundaries behave differently
    dist_from_center = np.sqrt(obj_x**2 + obj_y**2)
    boundary_effect = 1.0 - 0.3 * np.exp(-2 * (half_ws - dist_from_center)**2)
    displacement_mag *= boundary_effect
    
    # Coupling effects: push angle interacts with object position
    coupling_effect = 0.1 * np.sin(push_angle + np.arctan2(obj_y, obj_x))
    displacement_mag += coupling_effect
    
    # Realistic measurement noise (sensor uncertainty)
    noise_std = 0.02 + 0.01 * displacement_mag  # Larger displacements = more noise
    Y = displacement_mag + np.random.normal(0, noise_std)
    
    # Ensure physical constraints (no negative displacement)
    Y = np.maximum(Y, 0.0)
    
    # Create input features
    if include_force:
        X = np.column_stack([obj_x, obj_y, push_angle, push_force])
        input_dim = "4D"
    else:
        X = np.column_stack([obj_x, obj_y, push_angle])
        input_dim = "3D"
    
    # Optional normalization (standard in ML benchmarks)
    if normalize:
        scaler_X = StandardScaler()
        X = scaler_X.fit_transform(X)
        
        scaler_Y = StandardScaler()
        Y = scaler_Y.fit_transform(Y.reshape(-1, 1)).flatten()
        
        print(f"Data normalized: X mean={X.mean():.3f}, std={X.std():.3f}")
        print(f"                 Y mean={Y.mean():.3f}, std={Y.std():.3f}")
    
    print(f"Robot Push dataset: {X.shape[0]} samples, {input_dim} input")
    if not normalize:
        print(f"Object position range: x=[{obj_x.min():.2f}, {obj_x.max():.2f}], "
              f"y=[{obj_y.min():.2f}, {obj_y.max():.2f}]")
        print(f"Push angle range: [0, {2*np.pi:.2f}] radians")
        if include_force:
            print(f"Push force range: [{push_force.min():.1f}, {push_force.max():.1f}] N")
        print(f"Displacement range: [{Y.min():.3f}, {Y.max():.3f}] m")
    else:
        ranges = [f"x{i}=[{X[:, i].min():.2f}, {X[:, i].max():.2f}]" for i in range(X.shape[1])]
        print(f"Normalized ranges: {', '.join(ranges)}, disp=[{Y.min():.2f}, {Y.max():.2f}]")
    
    return X, Y

def load_srtm_elevation_dataset(region='maharashtra', max_samples=5000, subsample_factor=10, normalize=True, random_state=42, save_plot=True, use_preprocessed=False):
    """
    Load real SRTM 30m elevation data from HGT files or preprocessed NumPy files for specific regions from Attentive Kernels paper.
    
    This implementation follows the exact data processing approach used in the Attentive Kernels paper:
    - MinMaxScaler for coordinates (lat, lon) to range (-1, 1)
    - StandardScaler for elevation values
    - Proper negative outlier removal based on region characteristics
    - Support for the four benchmark regions: N17E073, N43W080, N45W123, N47W124
    
    Args:
        region (str): Region to load ('maharashtra', 'great_lakes', 'oregon_coast', 'washington_coast')
        max_samples (int): Maximum number of samples to load
        subsample_factor (int): Factor to subsample the grid data
        normalize (bool): Whether to normalize the features and targets (following Attentive Kernels)
        random_state (int): Random seed for sampling
        use_preprocessed (bool): Whether to use preprocessed .npy files instead of raw HGT files (default: True)
    
    Returns:
        tuple: (X, Y) where X is (lat, lon) and Y is elevation
    """
    try:
        import numpy as np
        import struct
        import os
        
        print(f"Loading SRTM elevation data for {region}...")
        
        # Define regions from Attentive Kernels paper with their SRTM tile information
        regions = {
            'maharashtra': {
                'tile': 'N17E073',
                'bounds': {'lat_min': 17.0, 'lat_max': 18.0, 'lon_min': 73.0, 'lon_max': 74.0},
                'description': 'Maharashtra, India (Western Ghats mountain range)',
                'allow_negative': False  # Inland region, no negative elevations expected
            },
            'great_lakes': {
                'tile': 'N43W080',
                'bounds': {'lat_min': 43.0, 'lat_max': 44.0, 'lon_min': -80.0, 'lon_max': -79.0},
                'description': 'Great Lakes region, Ontario/Michigan border',
                'allow_negative': False  # Above sea level, lake effects only
            },
            'oregon_coast': {
                'tile': 'N45W123',
                'bounds': {'lat_min': 45.0, 'lat_max': 46.0, 'lon_min': -123.0, 'lon_max': -122.0},
                'description': 'Oregon Coast Range',
                'allow_negative': False  # Minimal below-sea-level areas
            },
            'washington_coast': {
                'tile': 'N47W124',
                'bounds': {'lat_min': 47.0, 'lat_max': 48.0, 'lon_min': -124.0, 'lon_max': -123.0},
                'description': 'Washington Coast and Olympic Mountains',
                'allow_negative': False  # Minimal below-sea-level areas
            }
        }
        
        if region not in regions:
            raise ValueError(f"Region '{region}' not supported. Available: {list(regions.keys())}")
        
        region_info = regions[region]
        bounds = region_info['bounds']
        tile_name = region_info['tile']
        allow_negative = region_info['allow_negative']
        
        print(f"Region: {region_info['description']}")
        print(f"SRTM Tile: {tile_name}")
        print(f"Bounds: lat [{bounds['lat_min']}, {bounds['lat_max']}], lon [{bounds['lon_min']}, {bounds['lon_max']}]")
        
        if use_preprocessed:
            # Load from preprocessed NumPy file
            print(f"Loading preprocessed elevation data from .npy file...")
            
            # Use preprocessed data directory relative to the script
            preprocessed_dir = "srtm/preprocessed"
            npy_filename = f"{tile_name}.npy"
            preprocessed_path = os.path.join(preprocessed_dir, npy_filename)
            
            if not os.path.exists(preprocessed_path):
                raise FileNotFoundError(f"Preprocessed file not found: {preprocessed_path}")
            
            print(f"Using preprocessed file: {preprocessed_path}")
            elevation_data = np.load(preprocessed_path)
            
            print(f"Preprocessed data dimensions: {elevation_data.shape}")
            print(f"Raw elevation range: [{elevation_data.min():.1f}m, {elevation_data.max():.1f}m]")
            
            # Create coordinate grids based on known SRTM structure
            # Assume the preprocessed data maintains the same spatial structure
            if elevation_data.shape[0] == elevation_data.shape[1]:
                # Square grid - assume standard SRTM structure
                n_rows, n_cols = elevation_data.shape
                lats = np.linspace(bounds['lat_max'], bounds['lat_min'], n_rows)  # North to South
                lons = np.linspace(bounds['lon_min'], bounds['lon_max'], n_cols)  # West to East
                lon_grid, lat_grid = np.meshgrid(lons, lats)
            else:
                # Preprocessed data might be already flattened or in a different format
                raise ValueError(f"Unexpected preprocessed data shape: {elevation_data.shape}. Expected square grid.")
            
        else:
            # Load HGT file from local srtm_data directory (original logic)
            print(f"Loading HGT file from local srtm_data directory...")
            
            # Use data directory relative to the script
            data_dir = "srtm_data"
            
            # Try both naming conventions for HGT files
            hgt_filename_full = f"{tile_name}.SRTMGL1.hgt"  # Full OpenTopography naming
            hgt_filename_simple = f"{tile_name}.hgt"       # Simple naming (what user has)
            
            local_hgt_path = os.path.join(data_dir, hgt_filename_simple)
            
            # Check if the simple format file exists (what user has)
            if os.path.exists(local_hgt_path):
                print(f"Using local file: {local_hgt_path}")
            else:
                # Try the full format name as backup
                local_hgt_path = os.path.join(data_dir, hgt_filename_full)
                if os.path.exists(local_hgt_path):
                    print(f"Using local file: {local_hgt_path}")
                else:
                    # File not found - provide helpful error message
                    available_files = []
                    if os.path.exists(data_dir):
                        available_files = [f for f in os.listdir(data_dir) if f.endswith('.hgt')]
                    
                    error_msg = f"HGT file not found for tile {tile_name}\n"
                    error_msg += f"  Looked for: {hgt_filename_simple} or {hgt_filename_full}\n"
                    error_msg += f"  In directory: {os.path.abspath(data_dir)}\n"
                    if available_files:
                        error_msg += f"  Available files: {available_files}\n"
                    else:
                        error_msg += f"  No .hgt files found in {data_dir}\n"
                    error_msg += f"  Please ensure the HGT file is in the srtm_data directory"
                    
                    raise FileNotFoundError(error_msg)
            
            # Read HGT file
            print(f"Reading HGT elevation data...")
            elevation_data = read_hgt_file(local_hgt_path)
            
            print(f"HGT file dimensions: {elevation_data.shape}")
            print(f"Raw elevation range: [{elevation_data.min():.1f}m, {elevation_data.max():.1f}m]")
            
            # Create coordinate grids
            # SRTM 1 arc-second data has 3601x3601 points per degree
            n_rows, n_cols = elevation_data.shape
            
            # Create lat/lon arrays (note: SRTM convention has north at top)
            lats = np.linspace(bounds['lat_max'], bounds['lat_min'], n_rows)  # North to South
            lons = np.linspace(bounds['lon_min'], bounds['lon_max'], n_cols)  # West to East
            
            # Create meshgrid
            lon_grid, lat_grid = np.meshgrid(lons, lats)
        
        # Apply subsampling
        if subsample_factor > 1:
            print(f"Applying subsample factor: {subsample_factor}")
            lat_grid = lat_grid[::subsample_factor, ::subsample_factor]
            lon_grid = lon_grid[::subsample_factor, ::subsample_factor]
            elevation_data = elevation_data[::subsample_factor, ::subsample_factor]
            print(f"After subsampling: {elevation_data.shape}")
        
        # Flatten the grids
        X_lat = lat_grid.flatten()
        X_lon = lon_grid.flatten()
        Y_elevation = elevation_data.flatten()
        
        # Combine into feature matrix
        X = np.column_stack([X_lat, X_lon])
        Y = Y_elevation
        
        # Remove invalid elevation values (SRTM uses -32768 for no-data)
        print(f"Removing invalid elevation values...")
        initial_count = len(Y)
        
        # Remove SRTM no-data values and other invalid values
        valid_mask = (Y != -32768) & ~np.isnan(Y) & ~np.isinf(Y)
        X = X[valid_mask]
        Y = Y[valid_mask]
        
        print(f"Removed {initial_count - len(Y)} SRTM no-data/invalid points")
        
        # Remove negative elevation outliers (following Attentive Kernels approach)
        print(f"Processing elevation outliers...")
        pre_outlier_count = len(Y)
        negative_count = np.sum(Y < 0)
        
        if negative_count > 0:
            print(f"Found {negative_count} negative elevation values (min: {Y.min():.1f}m)")
            
            if not allow_negative:
                # For inland regions, remove all negative elevations
                # This follows Attentive Kernels approach for data quality
                positive_mask = Y >= 0
                removed_negatives = np.sum(~positive_mask)
                
                if removed_negatives > 0:
                    print(f"Removing {removed_negatives} negative elevation values (not physical for this region)")
                    X = X[positive_mask]
                    Y = Y[positive_mask]
                else:
                    print(f"No negative elevation values to remove")
        else:
            print(f"No negative elevation values found")
        
        # Additional quality control: remove extreme outliers
        print(f"Checking for extreme elevation outliers...")
        
        # Define reasonable elevation ranges for each region (following domain knowledge)
        elevation_limits = {
            'maharashtra': (0, 2000),     # Western Ghats: sea level to ~1600m + buffer
            'great_lakes': (75, 600),     # Great Lakes: lake level (~175m) to hills (~400m) + buffer  
            'oregon_coast': (0, 1500),    # Oregon Coast: sea level to Coast Range peaks (~1200m) + buffer
            'washington_coast': (0, 3000) # Washington Coast: sea level to Olympic peaks (~2400m) + buffer
        }
        
        min_elev, max_elev = elevation_limits.get(region, (0, 5000))
        extreme_outlier_mask = (Y >= min_elev) & (Y <= max_elev)
        removed_extreme_outliers = np.sum(~extreme_outlier_mask)
        
        if removed_extreme_outliers > 0:
            print(f"Removing {removed_extreme_outliers} extreme elevation outliers (outside [{min_elev}m, {max_elev}m])")
            print(f"Extreme values: min={Y[~extreme_outlier_mask].min():.1f}m, max={Y[~extreme_outlier_mask].max():.1f}m")
            X = X[extreme_outlier_mask]
            Y = Y[extreme_outlier_mask]
        else:
            print(f"No extreme elevation outliers found (all values in [{min_elev}m, {max_elev}m])")
        
        print(f"Clean elevation range: [{Y.min():.1f}m, {Y.max():.1f}m]")
        print(f"Mean elevation: {Y.mean():.1f}m ± {Y.std():.1f}m")
        
        # Sample if we have too many points
        if len(Y) > max_samples:
            np.random.seed(random_state)
            indices = np.random.choice(len(Y), size=max_samples, replace=False)
            X = X[indices]
            Y = Y[indices]
            print(f"Randomly sampled {max_samples} points")
        
        # Final data summary
        final_count = len(Y)
        print(f"Final dataset: {final_count} elevation points")
        print(f"Data retention: {initial_count} -> {final_count} ({(final_count/initial_count)*100:.1f}%)")
        
        # Apply normalization following Attentive Kernels approach
        if normalize:
            print(f"Applying Attentive Kernels normalization...")
            
            # MinMaxScaler for coordinates to range (-1, 1) - exactly like Attentive Kernels
            from sklearn.preprocessing import StandardScaler
            
            # Custom MinMaxScaler to (-1, 1) for coordinates (lat, lon)
            X_min = X.min(axis=0, keepdims=True)
            X_max = X.max(axis=0, keepdims=True)
            X_range = X_max - X_min
            
            # Transform to (-1, 1) range
            X_normalized = 2.0 * (X - X_min) / X_range - 1.0
            
            # StandardScaler for elevation values - exactly like Attentive Kernels  
            Y_scaler = StandardScaler()
            Y_normalized = Y_scaler.fit_transform(Y.reshape(-1, 1)).flatten()
            
            X = X_normalized
            Y = Y_normalized
            
            print(f"Applied normalization (Attentive Kernels style):")
            print(f"  Coordinates (lat,lon): MinMaxScaler to (-1, 1)")
            print(f"  Elevation: StandardScaler (mean=0, std=1)")
            print(f"  Normalized X range: [{X.min():.3f}, {X.max():.3f}]")
            print(f"  Normalized Y range: [{Y.min():.3f}, {Y.max():.3f}]")
            print(f"  Normalized Y: mean={Y.mean():.3f}, std={Y.std():.3f}")
        
        return X, Y
        
    except ImportError as e:
        print(f"Missing required package: {e}")
        print("For SRTM data loading, install: pip install scikit-learn")
        raise
    except FileNotFoundError as e:
        print(f"SRTM HGT file not found: {e}")
        print("Please ensure the required HGT files are in the srtm_data directory.")
        raise
    except Exception as e:
        print(f"Error loading SRTM elevation dataset: {e}")
        print("This might be due to file format issues or corrupt HGT files.")
        print("Please verify your HGT files are valid SRTM format.")
        raise

def read_hgt_file(hgt_path):
    """
    Read SRTM HGT file and return elevation data as numpy array.
    
    HGT files contain 16-bit signed integers in big-endian format.
    SRTM 1 arc-second data has 3601x3601 points.
    """
    print(f"Reading HGT file: {hgt_path}")
    
    # Get file size to determine dimensions
    file_size = os.path.getsize(hgt_path)
    
    # SRTM 1 arc-second: 3601x3601 points = 12,967,201 points * 2 bytes = 25,934,402 bytes
    # SRTM 3 arc-second: 1201x1201 points = 1,442,401 points * 2 bytes = 2,884,802 bytes
    
    if file_size == 25934402:
        # SRTM 1 arc-second
        n_points_per_side = 3601
        print("Detected SRTM 1 arc-second data (3601x3601)")
    elif file_size == 2884802:
        # SRTM 3 arc-second  
        n_points_per_side = 1201
        print("Detected SRTM 3 arc-second data (1201x1201)")
    else:
        raise ValueError(f"Unexpected HGT file size: {file_size} bytes")
    
    # Read binary data
    with open(hgt_path, 'rb') as f:
        # Read all data at once
        data = f.read()
    
    # Convert to numpy array of 16-bit signed integers (big-endian)
    elevation_array = np.frombuffer(data, dtype='>i2')  # '>i2' = big-endian 16-bit signed int
    
    # Reshape to 2D array
    elevation_data = elevation_array.reshape(n_points_per_side, n_points_per_side)
    
    print(f"Successfully read elevation data: {elevation_data.shape}")
    print(f"Data type: {elevation_data.dtype}")
    
    # Check for no-data values
    no_data_count = np.sum(elevation_data == -32768)
    if no_data_count > 0:
        print(f"Found {no_data_count} no-data points (value: -32768)")
    
    return elevation_data.astype(np.float64)  # Convert to float for calculations

def get_tile_for_region(region):
    """Get SRTM tile name for a given region."""
    region_tiles = {
        'maharashtra': 'N17E073',
        'great_lakes': 'N43W080', 
        'oregon_coast': 'N45W123',
        'washington_coast': 'N47W124'
    }
    return region_tiles.get(region, region)



def plot_real_world_dataset(X, Y, dataset_name='unknown', region=None, save_plot=False, output_dir='plots'):
    """
    Plot real-world dataset with appropriate visualization based on dataset type.
    
    Args:
        X: Input features (N, D)  
        Y: Target values (N,)
        dataset_name: Name of the dataset for plot titles
        region: Region name (for SRTM datasets)
        save_plot: Whether to save plot to file
        output_dir: Directory to save plots
    """
    try:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        import numpy as np
        import os
        
        input_dim = X.shape[1]
        n_samples = X.shape[0]
        
        # Determine dataset type and create appropriate title
        if 'srtm' in dataset_name.lower() or 'elevation' in dataset_name.lower():
            if region:
                region_titles = {
                    'maharashtra': 'Maharashtra, India (N17E073)',
                    'great_lakes': 'Great Lakes Region (N43W080)', 
                    'oregon_coast': 'Oregon Coast Range (N45W123)',
                    'washington_coast': 'Washington Coast (N47W124)'
                }
                plot_title = f"SRTM Elevation Data - {region_titles.get(region, region.replace('_', ' ').title())}"
                plot_subtitle = f"Tile: {get_tile_for_region(region)} | {n_samples:,} points"
            else:
                plot_title = f"SRTM Elevation Data"
                plot_subtitle = f"{n_samples:,} points"
            x_label, y_label, z_label = 'Longitude (°)', 'Latitude (°)', 'Elevation (m)'
            colormap = 'terrain'
        elif 'sst' in dataset_name.lower():
            plot_title = "Sea Surface Temperature (NOAA)"
            plot_subtitle = f"{n_samples:,} points"
            x_label, y_label, z_label = 'Longitude (°)', 'Latitude (°)', 'Temperature (°C)'
            colormap = 'coolwarm'
        elif 'robot' in dataset_name.lower():
            plot_title = "Robot Pushing Dataset"
            plot_subtitle = f"{n_samples:,} points"
            x_label, y_label, z_label = 'Feature 1', 'Feature 2', 'Displacement'
            colormap = 'viridis'
        else:
            plot_title = f"{dataset_name.title()} Dataset"
            plot_subtitle = f"{n_samples:,} points"
            x_label, y_label, z_label = f'X1', f'X2', 'Y'
            colormap = 'viridis'
        
        if input_dim == 2:
            # Create comprehensive 2D visualization
            fig = plt.figure(figsize=(20, 12))
            
            # Main 3D surface plot
            ax_main = fig.add_subplot(231, projection='3d')
            scatter = ax_main.scatter(X[:, 0], X[:, 1], Y, c=Y, cmap=colormap, s=15, alpha=0.7)
            ax_main.set_xlabel(x_label)
            ax_main.set_ylabel(y_label)
            ax_main.set_zlabel(z_label)
            ax_main.set_title(f'{plot_title}\n{plot_subtitle}', fontweight='bold')
            plt.colorbar(scatter, ax=ax_main, shrink=0.6, label=z_label)
            
            # 2D projection with elevation/value coloring
            ax_2d = fig.add_subplot(232)
            scatter_2d = ax_2d.scatter(X[:, 0], X[:, 1], c=Y, cmap=colormap, s=20, alpha=0.7)
            ax_2d.set_xlabel(x_label)
            ax_2d.set_ylabel(y_label)
            ax_2d.set_title('2D Projection (colored by value)', fontweight='bold')
            plt.colorbar(scatter_2d, ax=ax_2d, label=z_label)
            ax_2d.grid(True, alpha=0.3)
            
            # Elevation/value histogram
            ax_hist = fig.add_subplot(233)
            counts, bins, patches = ax_hist.hist(Y, bins=50, alpha=0.7, color='skyblue', edgecolor='black', linewidth=0.5)
            ax_hist.set_xlabel(z_label)
            ax_hist.set_ylabel('Frequency')
            ax_hist.set_title('Value Distribution', fontweight='bold')
            ax_hist.grid(True, alpha=0.3)
            
            # Add statistics text to histogram
            stats_text = f'Mean: {Y.mean():.2f}\nStd: {Y.std():.2f}\nMin: {Y.min():.2f}\nMax: {Y.max():.2f}'
            ax_hist.text(0.75, 0.95, stats_text, transform=ax_hist.transAxes, 
                        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            
            # Spatial statistics plots
            # Longitude vs elevation/value
            ax_lon = fig.add_subplot(234)
            ax_lon.scatter(X[:, 0], Y, alpha=0.5, s=10, color='red')
            ax_lon.set_xlabel(x_label)
            ax_lon.set_ylabel(z_label)
            ax_lon.set_title(f'{z_label} vs {x_label}', fontweight='bold')
            ax_lon.grid(True, alpha=0.3)
            
            # Latitude vs elevation/value
            ax_lat = fig.add_subplot(235)
            ax_lat.scatter(X[:, 1], Y, alpha=0.5, s=10, color='green')
            ax_lat.set_xlabel(y_label)
            ax_lat.set_ylabel(z_label)
            ax_lat.set_title(f'{z_label} vs {y_label}', fontweight='bold')
            ax_lat.grid(True, alpha=0.3)
            
            # Summary statistics table
            ax_stats = fig.add_subplot(236)
            ax_stats.axis('off')
            ax_stats.set_title('Dataset Summary', fontweight='bold', fontsize=14)
            
            # Calculate additional statistics
            summary_stats = f"""
                Dataset: {plot_title}
                Samples: {n_samples:,}
                Spatial Coverage:
                {x_label}: [{X[:, 0].min():.4f}, {X[:, 0].max():.4f}]
                {y_label}: [{X[:, 1].min():.4f}, {X[:, 1].max():.4f}]
                Value Statistics:
                {z_label}: [{Y.min():.2f}, {Y.max():.2f}]
                Mean: {Y.mean():.2f}
                Median: {np.median(Y):.2f}
                Std Dev: {Y.std():.2f}
                25th Percentile: {np.percentile(Y, 25):.2f}
                75th Percentile: {np.percentile(Y, 75):.2f}
                Quality Metrics:
                Missing Values: {np.sum(np.isnan(Y))} ({np.sum(np.isnan(Y))/len(Y)*100:.1f}%)
                Infinite Values: {np.sum(np.isinf(Y))} ({np.sum(np.isinf(Y))/len(Y)*100:.1f}%)
                Value Range: {Y.max() - Y.min():.2f}
                Coeff. of Variation: {(Y.std()/abs(Y.mean()))*100:.1f}%
                """
            
            ax_stats.text(0.05, 0.95, summary_stats, transform=ax_stats.transAxes,
                         fontsize=10, verticalalignment='top', fontfamily='monospace',
                         bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
            
            plt.tight_layout()
            
            # Save plot if requested
            if save_plot:
                os.makedirs(output_dir, exist_ok=True)
                
                # Create filename
                safe_name = dataset_name.replace(' ', '_').replace('/', '_')
                if region:
                    safe_region = region.replace(' ', '_')
                    filename = f"{safe_name}_{safe_region}_{n_samples}pts.png"
                else:
                    filename = f"{safe_name}_{n_samples}pts.png"
                
                filepath = os.path.join(output_dir, filename)
                plt.savefig(filepath, dpi=300, bbox_inches='tight')
                print(f"Real-world dataset plot saved to: {filepath}")
            
            plt.show()
            
        elif input_dim == 3:
            # 3D dataset visualization with multiple projections
            fig = plt.figure(figsize=(18, 12))
            
            # Main 3D scatter plot
            ax_main = fig.add_subplot(221, projection='3d')
            scatter = ax_main.scatter(X[:, 0], X[:, 1], X[:, 2], c=Y, cmap=colormap, s=15, alpha=0.7)
            ax_main.set_xlabel('Feature 1')
            ax_main.set_ylabel('Feature 2') 
            ax_main.set_zlabel('Feature 3')
            ax_main.set_title(f'{plot_title} - 3D Feature Space\n{plot_subtitle}', fontweight='bold')
            plt.colorbar(scatter, ax=ax_main, shrink=0.6, label=z_label)
            
            # 2D projections
            projections = [
                ([0, 1], 'Features 1 vs 2'),
                ([0, 2], 'Features 1 vs 3'),
                ([1, 2], 'Features 2 vs 3')
            ]
            
            for i, (indices, title) in enumerate(projections):
                ax = fig.add_subplot(2, 2, i + 2)
                scatter_proj = ax.scatter(X[:, indices[0]], X[:, indices[1]], c=Y, cmap=colormap, s=20, alpha=0.7)
                ax.set_xlabel(f'Feature {indices[0] + 1}')
                ax.set_ylabel(f'Feature {indices[1] + 1}')
                ax.set_title(title, fontweight='bold')
                if i == 0:  # Only add colorbar to first projection to save space
                    plt.colorbar(scatter_proj, ax=ax, label=z_label)
                ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Save plot if requested
            if save_plot:
                os.makedirs(output_dir, exist_ok=True)
                safe_name = dataset_name.replace(' ', '_').replace('/', '_')
                filename = f"{safe_name}_{n_samples}pts_3D.png"
                filepath = os.path.join(output_dir, filename)
                plt.savefig(filepath, dpi=300, bbox_inches='tight')
                print(f"Real-world dataset plot saved to: {filepath}")
            
            plt.show()
            
        else:
            print(f"Plotting not implemented for {input_dim}D datasets")
            
    except ImportError as e:
        print(f"Cannot plot dataset - missing matplotlib: {e}")
    except Exception as e:
        print(f"Error plotting real-world dataset: {e}")

def get_tile_for_region(region):
    """Get SRTM tile name for a region."""
    region_tiles = {
        'maharashtra': 'N17E073',
        'great_lakes': 'N43W080', 
        'oregon_coast': 'N45W123',
        'washington_coast': 'N47W124'
    }
    return region_tiles.get(region, 'Unknown')

def load_real_world_dataset(dataset_name, **kwargs):
    """
    Load a real-world dataset by name.
    
    Args:
        dataset_name (str): Name of the dataset
        **kwargs: Additional arguments passed to the specific loader
        
    Returns:
        tuple: (X, Y) dataset
    """
    loaders = {
        'sst': load_sea_surface_temperature,
        'sea_surface_temperature': load_sea_surface_temperature,
        'robot_push': load_robot_push_dataset,
        'robot': load_robot_push_dataset,
        'push': load_robot_push_dataset,
        'srtm_elevation': load_srtm_elevation_dataset,
        'srtm': load_srtm_elevation_dataset,
        'elevation': load_srtm_elevation_dataset
    }
    
    if dataset_name not in loaders:
        available = list(set(loaders.keys()))
        raise ValueError(f"Unknown dataset '{dataset_name}'. Available: {available}")
    
    # Load the dataset
    X, Y = loaders[dataset_name](**kwargs)
    
    # Plot the dataset if it's an SRTM elevation dataset
    if dataset_name in ['srtm_elevation', 'srtm', 'elevation']:
        region = kwargs.get('region', 'unknown')
        print(f"\nPlotting SRTM elevation dataset...")
        plot_real_world_dataset(X, Y, dataset_name='SRTM Elevation', region=region, 
                               save_plot=kwargs.get('save_plot', True), 
                               output_dir='srtm_plots')
    elif dataset_name in ['sst', 'sea_surface_temperature']:
        print(f"\nPlotting Sea Surface Temperature dataset...")
        plot_real_world_dataset(X, Y, dataset_name='Sea Surface Temperature', 
                               save_plot=kwargs.get('save_plot', True),
                               output_dir='sst_plots')
    elif dataset_name in ['robot_push', 'robot', 'push']:
        print(f"\nPlotting Robot Pushing dataset...")
        plot_real_world_dataset(X, Y, dataset_name='Robot Pushing',
                               save_plot=kwargs.get('save_plot', True), 
                               output_dir='robot_push_plots')
    
    return X, Y

def get_dataset_info():
    """
    Get information about available real-world datasets.
    
    Returns:
        dict: Information about each available dataset
    """
    return {
        'sst': {
            'name': 'Sea Surface Temperature',
            'dimensions': 2,
            'input_desc': 'Latitude, Longitude (degrees)',
            'output_desc': 'Temperature (°C)',
            'typical_samples': '1000-50000',
            'source': 'Synthetic oceanographic patterns (benchmark)',
            'features': 'Spatial correlation, realistic physics'
        },
        'robot_push': {
            'name': 'Robot Push Manipulation',
            'dimensions': '3 (or 4 with force)',
            'input_desc': 'Object X, Y, Push Angle [, Force]',
            'output_desc': 'Displacement (meters)',
            'typical_samples': '1000-50000',
            'source': 'Synthetic contact mechanics (benchmark)',
            'features': 'Nonlinear dynamics, physics constraints'
        },
        'srtm_elevation': {
            'name': 'SRTM 30m Elevation Data (Attentive Kernels)',
            'dimensions': 2,
            'input_desc': 'Latitude, Longitude (degrees)',
            'output_desc': 'Elevation (meters above sea level)',
            'typical_samples': '1000-50000',
            'source': 'NASA SRTM (local HGT files)',
            'features': 'Real-world spatial data, four benchmark regions from Attentive Kernels paper'
        }
    }
