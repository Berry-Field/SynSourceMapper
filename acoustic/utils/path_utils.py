import os
import tempfile

def get_project_root():
    """
    Get the absolute path to the project root directory.
    
    Returns:
        str: Absolute path to the project root
    """
    # Return the directory where the acoustic package is located
    return os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(__file__)), '..'))

def get_data_path(relative_path=None):
    """
    Get the absolute path to the data directory or a file within it.
    
    Args:
        relative_path (str, optional): Relative path within the data directory. Defaults to None.
        
    Returns:
        str: Absolute path to the data directory or the specified file
    """
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
    
    if relative_path is None:
        return data_dir
    
    return os.path.join(data_dir, relative_path)

def get_arrays_path(relative_path=None):
    """
    Get the absolute path to the arrays directory or a file within it.
    
    Args:
        relative_path (str, optional): Relative path within the arrays directory. Defaults to None.
        
    Returns:
        str: Absolute path to the arrays directory or the specified file
    """
    arrays_dir = os.path.join(get_data_path(), 'arrays')
    
    if relative_path is None:
        return arrays_dir
    
    return os.path.join(arrays_dir, relative_path)

def get_cache_dir(create_if_missing=True, custom_cache_dir=None):
    """
    Get the directory for caching computation results.
    
    Args:
        create_if_missing (bool, optional): Create the directory if it doesn't exist. Defaults to True.
        custom_cache_dir (str, optional): Custom cache directory path. Defaults to None.
        
    Returns:
        str: Absolute path to the cache directory
    """
    # Use custom cache directory if provided
    if custom_cache_dir is not None:
        cache_dir = custom_cache_dir
    else:
        # Otherwise use default path in system temp directory
        cache_dir = os.path.join(tempfile.gettempdir(), 'acoustic_cache')
    
    # Create if it doesn't exist
    if create_if_missing and not os.path.exists(cache_dir):
        os.makedirs(cache_dir, exist_ok=True)
    
    return cache_dir

def get_A_matrix_cache_path(config_params):
    """
    Get the path for caching A matrices based on configuration parameters.
    
    Args:
        config_params (dict): Dictionary containing configuration parameters
        
    Returns:
        str: Path where the A matrix should be cached
    """
    # Check for custom cache directory
    custom_cache_dir = config_params.get('cache_dir', None)
    
    # Extract parameters
    if_ref_distance_1 = config_params.get('if_ref_distance_1', False)
    c = config_params.get('c', 343)
    x_grid = config_params.get('x_grid', 101)
    y_grid = config_params.get('y_grid', 101)
    x_range = config_params.get('x_range', (-50, 50))
    y_range = config_params.get('y_range', (-50, 50))
    z_scan = config_params.get('z_scan', 50)
    fs = config_params.get('fs', 16000)
    act_bin = config_params.get('act_bin', 0)
    nfft = config_params.get('nfft', 1024)
    act_freq = config_params.get('act_freq', 0.0)
    
    # Create path components
    if_ref_distance_1_flag = "ref_1" if if_ref_distance_1 else "ref_source"
    
    # Create the directory path
    path = os.path.join(
        get_cache_dir(custom_cache_dir=custom_cache_dir),
        f'A_TMP_{if_ref_distance_1_flag}',
        f'c_{c}',
        f'x_grid_{x_grid}',
        f'y_grid_{y_grid}',
        f'x_range_{x_range[0]}_{x_range[1]}',
        f'y_range_{y_range[0]}_{y_range[1]}',
        f'z_scan_{z_scan}',
        f'fs_{fs}',
        f'act_bin_{act_bin}_of_nfft_{nfft}_act_freq_{act_freq:.1f}'
    )
    
    # Ensure directory exists
    os.makedirs(path, exist_ok=True)
    
    return os.path.join(path, 'A_matrix.pt') 