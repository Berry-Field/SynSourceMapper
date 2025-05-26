import numpy as np
import scipy.io as sio
import os
import shutil
from acoustic.utils.path_utils import get_arrays_path, get_data_path, get_project_root

class ArrayPositions:
    """Class for array positions, used to get 3D coordinates of different array elements"""
    
    def __init__(self, array_id):
        """
        Initialize the array positions class
        
        Args:
            array_id: array_id
        """
        self.array_id = array_id
        
    def _find_array_file(self, file_name):
        """
        Find the array data file
        
        Args:
            file_name: file_name
            
        Returns:
            str: file_path
        """
        # Check possible locations
        possible_locations = [
            get_arrays_path(file_name),             # acoustic/data/arrays/
            get_data_path(file_name),               # acoustic/data/
            os.path.join(get_project_root(), 'arraybox', file_name),  # project_root/arraybox/
            os.path.join(os.path.dirname(get_project_root()), 'arraybox', file_name)  # parent_of_project_root/arraybox/
        ]
        
        for location in possible_locations:
            if os.path.exists(location):
                return location
                
        # If file not found, try to copy the first .mat file found and rename it
        found_mat_files = []
        for root_dir in [get_data_path(), get_project_root(), os.path.dirname(get_project_root())]:
            for root, _, files in os.walk(root_dir):
                for file in files:
                    if file.endswith('.mat') and 'array' in file.lower():
                        found_mat_files.append(os.path.join(root, file))
        
        if found_mat_files:
            # Ensure target directory exists
            os.makedirs(get_arrays_path(), exist_ok=True)
            target_file = get_arrays_path(file_name)
            # Copy the first file found
            shutil.copy2(found_mat_files[0], target_file)
            print(f"Found array file {found_mat_files[0]}, copied to {target_file}")
            return target_file
            
        return None
        
    def get_positions(self):
        """
        Get the 3D coordinates of array elements
        
        Returns:
            array_positions: Array element positions, shape (n_channels, 3)
        """
        if self.array_id == 1:
            # Find array file
            array_file = self._find_array_file('Tj64array.mat')
            
            if array_file is None:
                raise FileNotFoundError("Cannot find array file Tj64array.mat, please ensure the file exists in the data directory")
                
            print(f"Loading array data from: {array_file}")
            mat_data = sio.loadmat(array_file)
            # Get all variable names
            var_names = mat_data.keys()
            # Exclude system variables (variables starting with __)
            array_var_names = [name for name in var_names if not name.startswith('__')]
            
            if len(array_var_names) == 0:
                raise ValueError("No variables found in the mat file")
            elif len(array_var_names) > 1:
                print(f"Warning: mat file contains multiple variables: {array_var_names}, using the first one")
                
            array_positions = mat_data[array_var_names[0]]
            return array_positions
            
        elif self.array_id == 2:
            # Implement other array types
            pass
            
        else:
            raise ValueError(f"Unsupported array ID: {self.array_id}")
