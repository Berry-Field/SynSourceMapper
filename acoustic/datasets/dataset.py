import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import os

from acoustic.beamforming import (
    ConventionalBeamformingTorch,
    index_to_position, 
    calculate_noise_scm_formula, 
    get_fb_map, 
    create_blurred_image
)
from acoustic.data import ArrayPositions
from acoustic.utils import freq2fbin, fbin2freq, normalize_tensor, cal_A_matrix

class AcousticArrayDataset(Dataset):
    def __init__(self, config: dict, device: torch.device, stack_flag: bool = False):
        """Initialize acoustic array dataset
        
        Args:
            config (dict): Configuration dictionary
            device (torch.device): Computing device
            stack_flag (bool, optional): Whether to stack samples. Defaults to False.
        """
        self.device = device
        self.stack_flag = stack_flag
        self.config = config
        self._initialize_scanning_parameters()
        self._initialize_array_parameters()
        self._initialize_source_parameters()
        self._initialize_target_parameters()
        self._initialize_caches()
        self.rng = None
        
    def _initialize_scanning_parameters(self):
        """Initialize scanning parameters"""
        self.fs = self.config['scan_set']['fs']
        self.nfft = self.config['scan_set']['nfft']
        self.c = self.config['scan_set']['c']
        self.x_range = self.config['scan_set']['x_range']
        self.y_range = self.config['scan_set']['y_range']
        self.z_scan = self.config['scan_set']['z_scan']
        self.x_grid = self.config['scan_set']['x_grid']
        self.y_grid = self.config['scan_set']['y_grid']
        self.nob = int(self.x_grid * self.y_grid)
        self.x_resolution = (self.x_range[1] - self.x_range[0]) / (self.x_grid - 1)
        self.y_resolution = (self.y_range[1] - self.y_range[0]) / (self.y_grid - 1)
        
        # Get cache directory if specified
        self.cache_dir = self.config.get('cache_dir', None)
        if self.cache_dir:
            # Create the cache directory if it doesn't exist
            os.makedirs(self.cache_dir, exist_ok=True)
            print(f"Using custom A_matrix cache directory: {self.cache_dir}")
        
    def _initialize_array_parameters(self):
        """Initialize array parameters"""
        self.array_positions = ArrayPositions(self.config['scan_set']['array_id']).get_positions()
        
    def _initialize_source_parameters(self):
        """Initialize source parameters"""
        self.source_frequencies = self.config['source_set']['freq']
        self.minEnergy = self.config["source_set"]["minEnergy"]
        self.min_source_distance = self.config['source_set']['min_source_distance']  
        self.nos = self.config['source_set']['nos']  
        self.num_samples = self.config['source_set']['num_samples']
        self.noise_level = self.config['noise_level']
        
        # Check if nos and freq are lists
        if not isinstance(self.nos, list):
            raise ValueError("nos must be a list type")
            
    def _initialize_target_parameters(self):
        """Initialize target parameters"""
        self.blur = self.config['target_map_set']['blur']
        self.blur_level = self.config['target_map_set']['blur_level']
        self.norm_range = self.config['norm_range']
        self.use_FB = self.config['use_FB']
        
        if self.use_FB:
            print(f"Additional output: use_FB: {self.use_FB}\n")
            
    def _initialize_caches(self):
        """Initialize caches"""
        self.act_freqs = []
        self.A_cache = {}
        self.ws_cache = {}
        self.dvecs_cache = {}
        self.noise_scm_formula_cache = {}
        
        # Convert frequencies and initialize A matrix cache
        for freq in self.source_frequencies:
            act_bin = freq2fbin(freq, self.fs, self.nfft)
            act_freq = fbin2freq(act_bin, self.fs, self.nfft)
            self.act_freqs.append(act_freq)
            
        for freq in self.act_freqs:
            self._calculate_cache_for_frequency(freq)
            
        self.Nch = self.ws_cache[self.act_freqs[0]].shape[1]
        
        if not isinstance(self.act_freqs, list):
            raise ValueError("act_freqs must be a list type")
            
    def _calculate_cache_for_frequency(self, freq):
        """Calculate cache for specific frequency
        
        Args:
            freq (float): Frequency
        """
        # Initialize CB object to get ws and dvecs
        CB = ConventionalBeamformingTorch(
            self.array_positions, array_signals=None, fs=self.fs,
            x_range=self.x_range, y_range=self.y_range,
            x_grid=self.x_grid, y_grid=self.y_grid, z_scan=self.z_scan,
            c=self.c, nfft=self.nfft, noverlap=None,
            ana_freq=freq, ref_pressure=None,
            print_flag=False, device=self.device
        )
        ws, dvecs = CB.get_steering_vectors()
        
        # Build configuration parameters including cache path
        config_params = {
            'c': self.c,
            'x_grid': self.x_grid,
            'y_grid': self.y_grid,
            'x_range': self.x_range,
            'y_range': self.y_range,
            'z_scan': self.z_scan,
            'fs': self.fs,
            'nfft': self.nfft,
            'ana_freq': freq,
            'cache_dir': self.cache_dir  # Add cache_dir to config
        }
        
        # Calculate A matrix
        A = cal_A_matrix(
            ws, dvecs, self.array_positions, fs=self.fs, 
            x_range=self.x_range, y_range=self.y_range, 
            x_grid=self.x_grid, y_grid=self.y_grid, z_scan=self.z_scan,
            c=self.c, nfft=self.nfft, 
            ana_freq=freq, 
            print_flag=True,
            device=self.device,
            config_params=config_params  # Pass complete config with cache_dir
        )
        
        # Store caches
        self.A_cache[freq] = A
        self.ws_cache[freq] = ws.to(self.device)
        self.dvecs_cache[freq] = dvecs.to(self.device)
        self.noise_scm_formula_cache[freq] = calculate_noise_scm_formula(ws, self.device)

    def __len__(self) -> int:
        """Return dataset length
        
        Returns:
            int: Dataset length
        """
        return self.num_samples
    
    def set_seed(self, seed: int) -> None:
        """Set random seed
        
        Args:
            seed (int): Random seed
        """
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        
    def _get_random_num_points(self) -> int:
        """Generate number of points based on Rayleigh distribution
        
        Returns:
            int: Number of points from Rayleigh distribution with mean=2, max=5
        """
        # Generate from Rayleigh distribution with scale parameter to achieve mean=2
        # For Rayleigh distribution, mean = scale * sqrt(pi/2)
        # So scale = mean / sqrt(pi/2)
        scale = 2 / np.sqrt(np.pi/2)
        
        # Generate a random value from Rayleigh distribution
        value = self.rng.rayleigh(scale)
        
        # Convert to integer and clip to range [1, 5]
        num_points = min(max(int(round(value)), 1), 5)
        
        return num_points
    
    def _get_random_freq(self) -> float:
        """Randomly select frequency from freq list

        Returns:
            float: Randomly selected frequency
        """
        return self.rng.choice(self.act_freqs)

    def _get_source_positions(self, num_points: int) -> tuple:
        """Get random number of source positions
        
        Args:
            num_points (int): Number of sources
            
        Returns:
            tuple: (source_ac_position_index, source_energies, source_positions, image)
        """
        source_ac_position_index = np.zeros((2, num_points))
        source_energies = np.zeros((1, num_points))
        image = np.zeros((self.x_grid, self.y_grid))
        selected_points = []
        
        for i in range(num_points):
            while True:
                x = self.rng.integers(0, self.x_grid)
                y = self.rng.integers(0, self.y_grid)
                
                # Check distance to existing points
                valid_point = True
                for px, py in selected_points:
                    if abs(x - px) < self.min_source_distance and abs(y - py) < self.min_source_distance:
                        valid_point = False
                        break
                        
                if valid_point:
                    selected_points.append((x, y))
                    value = self.rng.uniform(self.minEnergy, 1.0)
                    image[x, y] = value
                    # Store source position
                    source_ac_position_index[0, i] = x
                    source_ac_position_index[1, i] = y
                    source_energies[0, i] = value
                    break
                    
        source_positions = np.zeros((2, num_points))
        for i in range(num_points):
            x, y = index_to_position(
                source_ac_position_index[0, i], 
                source_ac_position_index[1, i],
                self.x_range,
                self.y_range,
                self.x_grid,
                self.y_grid
            )
            source_positions[0, i] = x
            source_positions[1, i] = y

        return source_ac_position_index, source_energies, source_positions, image
    
    def _get_scm_wo_signal(self, source_ac_position_index: np.ndarray, source_energies: np.ndarray, analysis_f: float) -> torch.Tensor:
        """Get spatial correlation matrix (without using signal)

        Args:
            source_ac_position_index (np.ndarray): Source position index, shape (2, N)
            source_energies (np.ndarray): Source energies, shape (1, N)
            analysis_f (float): Analysis frequency
            
        Returns:
            torch.Tensor: Spatial correlation matrix
        """
        # Convert 2D index to 1D index
        source_indices = source_ac_position_index[1] * self.x_grid + source_ac_position_index[0]
        source_indices = source_indices.astype(int)
        
        # Get steering vectors for corresponding frequency
        dvecs = self.dvecs_cache[analysis_f]
        
        # Extract steering vectors corresponding to source positions
        source_dvecs = dvecs[source_indices]
        
        # Convert source_energies to diagonal matrix and convert to complex
        source_energies_complex = torch.from_numpy(source_energies[0]).to(source_dvecs.device).to(torch.complex64)
        source_energies_diag = torch.diag(source_energies_complex)
        
        # Calculate spatial correlation matrix
        scm = torch.matmul(source_dvecs.conj().T, torch.matmul(source_energies_diag, source_dvecs))
        return scm
    
    def _apply_normalization(self, x_tensor, y_tensor, target_map_tensor, PSF_tensor, scm_tensor, FB_map_tensor=None):
        """Apply normalization
        
        Args:
            x_tensor (torch.Tensor): Original image tensor
            y_tensor (torch.Tensor): DAS result tensor
            target_map_tensor (torch.Tensor): Target image tensor
            PSF_tensor (torch.Tensor): PSF tensor
            scm_tensor (torch.Tensor): Spatial correlation matrix
            FB_map_tensor (torch.Tensor, optional): FB beamforming result. Defaults to None.
            
        Returns:
            tuple: Normalized tensors
        """
        if self.norm_range == "None":
            if FB_map_tensor is not None:
                return x_tensor, y_tensor, target_map_tensor, PSF_tensor, scm_tensor, FB_map_tensor
            return x_tensor, y_tensor, target_map_tensor, PSF_tensor, scm_tensor
            
        tmp_value = y_tensor.max()
        x_tensor = x_tensor / tmp_value
        y_tensor = y_tensor / tmp_value
        target_map_tensor = target_map_tensor / tmp_value
        
        if self.norm_range != [0, 1]:
            target_map_tensor = normalize_tensor(target_map_tensor, self.norm_range)
            x_tensor = normalize_tensor(x_tensor, self.norm_range)
            y_tensor = normalize_tensor(y_tensor, self.norm_range)
            PSF_tensor = normalize_tensor(PSF_tensor, self.norm_range)
            
        scm_tensor = scm_tensor / tmp_value
        FB_map_tensor = FB_map_tensor / tmp_value
        
        if FB_map_tensor is not None:
            if self.norm_range == [-1, 1]:
                FB_map_tensor = normalize_tensor(FB_map_tensor, self.norm_range)
            elif self.norm_range != [0, 1]:
                raise ValueError(f"norm_range: {self.norm_range} is not supported for calculating FB beamforming results")
            return x_tensor, y_tensor, target_map_tensor, PSF_tensor, scm_tensor, FB_map_tensor
            
        return x_tensor, y_tensor, target_map_tensor, PSF_tensor, scm_tensor
    
    def _pad_source_data(self, source_ac_position_index, source_positions, source_energies):
        """Pad source data to maximum length
        
        Args:
            source_ac_position_index (np.ndarray): Source position index
            source_positions (np.ndarray): Source positions
            source_energies (np.ndarray): Source energies
            
        Returns:
            tuple: Padded data
        """
        max_nos = max(self.nos)
        
        # Extend source_ac_position_index to maximum length
        padded_position_index = np.full((2, max_nos), -999, dtype=np.float32)
        padded_position_index[:, :source_ac_position_index.shape[1]] = source_ac_position_index
        
        # Extend source_positions to maximum length
        padded_positions = np.full((2, max_nos), -999, dtype=np.float32)
        padded_positions[:, :source_positions.shape[1]] = source_positions
        
        # Extend source_energies to maximum length
        padded_energies = np.full((1, max_nos), -999, dtype=np.float32) 
        padded_energies[:, :source_energies.shape[1]] = source_energies
        
        return padded_position_index, padded_positions, padded_energies
    
    def __getitem__(self, idx):
        """Get a sample from dataset
        
        Args:
            idx (int): Sample index
            
        Returns:
            tuple: Data sample
        """
        if self.rng is None:
            raise ValueError("Random seed not set, please call set_seed method first")
        
        # 1. Randomly select frequency and number of points
        analysis_f = self._get_random_freq()
        freq_idx = self.act_freqs.index(analysis_f)
        source_f = self.source_frequencies[freq_idx]
        num_points = self._get_random_num_points()
        
        # 2. Get source positions and image
        source_ac_position_index, source_energies, source_positions, image = self._get_source_positions(num_points)
        scm_tensor = self._get_scm_wo_signal(source_ac_position_index, source_energies, analysis_f)
        
        # 3. Save original image
        original_image = image.copy()
        
        # 4. Apply blur effect (if needed)
        blurred_image = original_image
        if self.blur:
            blurred_image = create_blurred_image(
                original_image, 
                self.x_grid, 
                self.y_grid, 
                self.x_resolution, 
                self.y_resolution, 
                self.blur_level
            )
        
        # 5. Get A matrix and calculate DAS result
        A_matrix = self.A_cache[analysis_f]
        image_vector = image.reshape(-1, 1, order='F')
        image_vector_gpu = torch.from_numpy(image_vector).float().to(A_matrix.device)
        result_vector_gpu = torch.sparse.mm(A_matrix, image_vector_gpu.to_sparse()).squeeze(0)
        
        # 6. Reshape result
        y = result_vector_gpu.reshape(self.y_grid, self.x_grid).T
        
        # 7. Create return tensors
        target_map_tensor = torch.from_numpy(blurred_image).float().unsqueeze(0)
        x_tensor = torch.from_numpy(original_image).float().unsqueeze(0)
        y_tensor = y.unsqueeze(0).to(x_tensor.device)
        analysis_f_tensor = torch.tensor(analysis_f, dtype=torch.float)
        source_f_tensor = torch.tensor(source_f, dtype=torch.float)
        nos_tensor = torch.tensor(num_points, dtype=torch.long)
        
        # 8. Get PSF tensor
        middle_row = A_matrix[A_matrix.shape[0] // 2]
        middle_row_reshaped = middle_row.reshape(self.y_grid, self.x_grid).T
        PSF_tensor = middle_row_reshaped.unsqueeze(0)
        
        # 9. Add noise
        SNR = self.rng.uniform(self.noise_level[0], self.noise_level[1])
        source_energy = np.sum(np.abs(source_energies))
        noise_scm_formula = self.noise_scm_formula_cache[analysis_f].reshape(self.y_grid, self.x_grid).T
        noise_energy = source_energy / (10 ** (SNR/10) * self.nfft)
        y_tensor = y_tensor + noise_energy * noise_scm_formula.unsqueeze(0).to(y_tensor.device)
        scm_tensor = scm_tensor + torch.eye(self.Nch, dtype=torch.cfloat, device=scm_tensor.device) * (noise_energy*self.Nch*1 + 0j)
        
        # 10. Calculate FB beamforming result (if needed)
        FB_map_tensor = None
        if self.use_FB:
            FB_map = get_fb_map(scm_tensor, self.ws_cache[analysis_f], self.Nch)
            FB_map_tensor = FB_map.reshape(self.y_grid, self.x_grid).T
            FB_map_tensor = FB_map_tensor.unsqueeze(0).to(x_tensor.device)
        
        # 11. Normalize
        if self.use_FB:
            x_tensor, y_tensor, target_map_tensor, PSF_tensor, scm_tensor, FB_map_tensor = self._apply_normalization(
                x_tensor, y_tensor, target_map_tensor, PSF_tensor, scm_tensor, FB_map_tensor
            )
        else:
            x_tensor, y_tensor, target_map_tensor, PSF_tensor, scm_tensor = self._apply_normalization(
                x_tensor, y_tensor, target_map_tensor, PSF_tensor, scm_tensor
            )
        
        # 12. Stack samples (if needed)
        if self.stack_flag:
            source_ac_position_index, source_positions, source_energies = self._pad_source_data(
                source_ac_position_index, source_positions, source_energies
            )
        
        # 13. Return results
        if self.use_FB:
            return target_map_tensor, nos_tensor, analysis_f_tensor, source_f_tensor, A_matrix, y_tensor, PSF_tensor, source_ac_position_index, source_positions, x_tensor, source_energies, scm_tensor, FB_map_tensor
        else:
            return target_map_tensor, nos_tensor, analysis_f_tensor, source_f_tensor, A_matrix, y_tensor, PSF_tensor, source_ac_position_index, source_positions, x_tensor, source_energies, scm_tensor


def get_acoustic_array_dataloader(config: dict, device: torch.device, batch_size: int, num_workers: int, shuffle: bool, stack_flag: bool = False) -> DataLoader:
    """Get data loader for acoustic array dataset
    
    Args:
        config (dict): Configuration dictionary
        device (torch.device): Computing device
        batch_size (int): Batch size
        num_workers (int): Number of worker threads
        shuffle (bool): Whether to shuffle data
        stack_flag (bool, optional): Whether to stack samples. Defaults to False.
        
    Returns:
        DataLoader: Data loader
    """
    dataset = AcousticArrayDataset(config, device, stack_flag)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=False
    )
    return dataloader 