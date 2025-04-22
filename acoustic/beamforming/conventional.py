import numpy as np
import scipy.signal as signal
import torch
from acoustic.utils import fbin2freq, freq2fbin

class ConventionalBeamformingTorch:
    def __init__(self, array_positions: np.ndarray, array_signals: np.ndarray = None, fs: int = 16000,
                 x_range: tuple = (-50, 50), y_range: tuple = (-50, 50),
                 x_grid: int = 101, y_grid: int = 101, z_scan: int = 50,
                 c: int = 343, nfft: int = 1024, noverlap: int = None,
                 ana_freq: int = 100, ref_pressure: float = 2e-5,
                 print_flag: bool = False, device: torch.device = None, if_ref_distance_1: bool = False):
        """
        Initialize PyTorch-based Conventional Beamforming (CBF) algorithm
        
        Args:
            array_positions: Array element positions, shape (n_channels, 3)
            array_signals: Multi-channel signal data, shape (n_channels, n_samples)
            fs: Sampling rate, Hz
            x_range: X-axis scanning range (min, max), meters
            y_range: Y-axis scanning range (min, max), meters
            x_grid: Number of grid divisions on X-axis
            y_grid: Number of grid divisions on Y-axis
            z_scan: Scanning plane height, meters
            c: Speed of sound, m/s
            nfft: FFT points for STFT
            noverlap: Overlap points for STFT, default is nfft//2
            ana_freq: Analysis frequency, Hz
            ref_pressure: Reference sound pressure, Pa
            print_flag: Whether to print intermediate process information
            device: Computing device, 'cuda' or 'cpu'
        """
        self.device = device
        
        # Basic parameters
        self.array_positions = torch.tensor(array_positions, device=device)
        self.array_signals = torch.tensor(array_signals, device=device) if array_signals is not None else None
        self.fs = fs
        self.x_range = x_range
        self.y_range = y_range
        self.x_grid = x_grid
        self.y_grid = y_grid
        self.z_scan = z_scan
        self.c = c
        self.nfft = nfft
        self.noverlap = nfft//2 if noverlap is None else noverlap
        self.ana_freq = ana_freq
        self.ref_pressure = ref_pressure
        self.print_flag = print_flag
        self.if_ref_distance_1 = if_ref_distance_1
        # Calculate grid points
        self._setup_grid()
        
        # Initialize result storage
        self.map_cbf = None
        self.map_cbf_db = None
        self.ws = None
        self.dvecs = None
        self.SCM = None
        self.stft_data = None
        
        # Calculate frequency-related parameters
        self.act_bin = freq2fbin(self.ana_freq, self.fs, self.nfft)
        self.act_freq = fbin2freq(self.act_bin, self.fs, self.nfft)
        
    def _setup_grid(self) -> None:
        """Set up scanning grid"""
        x_points = torch.linspace(self.x_range[0], self.x_range[1], self.x_grid, device=self.device)
        y_points = torch.linspace(self.y_range[0], self.y_range[1], self.y_grid, device=self.device)
        X, Y = torch.meshgrid(x_points, y_points, indexing='xy')
        
        self.scan_points = torch.zeros((self.x_grid*self.y_grid, 3), device=self.device)
        self.scan_points[:, 0] = X.flatten()
        self.scan_points[:, 1] = Y.flatten()
        self.scan_points[:, 2] = self.z_scan
        
        if self.print_flag:
            print(f"Scanning grid setup complete: {self.x_grid}x{self.y_grid} points")
            print(f"Scanning range: X={x_points[0].item()}~{x_points[-1].item()}m, Y={y_points[0].item()}~{y_points[-1].item()}m, Z={self.z_scan}m")
            
    def _compute_stft(self, use_torch: bool = False) -> None:
        """Compute STFT"""
        if self.stft_data is not None:
            return
                
        window = signal.windows.hann(self.nfft)
        n_channels = self.array_positions.shape[0]
        
        _, _, Zxx = signal.stft(self.array_signals[0].cpu().numpy(), fs=self.fs, window=window,
                            nperseg=self.nfft, noverlap=self.noverlap,
                            nfft=self.nfft, return_onesided=True)
        n_freq_bins, n_time_frames = Zxx.shape
        stft_data = np.zeros((n_freq_bins, n_time_frames, n_channels), dtype=complex)
        
        for ch in range(n_channels):
            _, _, stft_data[:,:,ch] = signal.stft(self.array_signals[ch].cpu().numpy(), fs=self.fs,
                                                window=window, nperseg=self.nfft,
                                                noverlap=self.noverlap, nfft=self.nfft,
                                                return_onesided=True)
        
        self.stft_data = torch.tensor(stft_data, dtype=torch.complex64, device=self.device)
                                                  
    def _compute_scm(self) -> None:
        """Compute spatial correlation matrix"""
        if self.SCM is not None:
            return
            
        self._compute_stft()
        Y_cal = self.stft_data[self.act_bin, :, :]  # (time_frames, channels)
        Y_cal_T = Y_cal.T
        self.SCM = torch.matmul(Y_cal_T.conj(), Y_cal) / Y_cal.shape[0]
        self.SCM = self.SCM.to(device=self.device)
        
    def _compute_steering_vectors(self) -> None:
        """Compute steering vectors"""
        if self.ws is not None and self.dvecs is not None:
            return
        
        n_scan_points = self.scan_points.shape[0]
        n_channels = self.array_positions.shape[0]
        self.ws = torch.zeros((n_scan_points, n_channels), dtype=torch.complex64, device=self.device)
        self.dvecs = torch.zeros((n_scan_points, n_channels), dtype=torch.complex64, device=self.device)
        
        for i in range(n_scan_points):
            x, y, z = self.scan_points[i]
            w, dvec = get_steering_vector(x, y, z, self.ana_freq, self.c, self.array_positions, 
                                         if_torch=True, if_ref_distance_1=self.if_ref_distance_1)
            self.ws[i] = w
            self.dvecs[i] = dvec
            
    def _compute_cbf_map(self, ws: torch.Tensor = None, dvecs: torch.Tensor = None) -> None:
        """Calculate CBF result map
        
        Args:
            ws: Steering vectors, if None, use internally calculated results
            dvecs: Delay vectors, if None, use internally calculated results
        """
        self._compute_scm()
        
        if ws is None or dvecs is None:
            self._compute_steering_vectors()
            ws = self.ws
            dvecs = self.dvecs
            
        n_scan_points = self.scan_points.shape[0]
        n_channels = self.array_positions.shape[0]

        cbf_values = torch.zeros((n_scan_points,), device=self.device)
        for i in range(n_scan_points):
            cbf_values[i] = ws[i, :] @ self.SCM @ ws[i, :].conj() / (n_channels ** 2)
        
        # Reshape to grid shape
        self.map_cbf = cbf_values.reshape(self.y_grid, self.x_grid)
        self.map_cbf_db = 20 * torch.log10(torch.sqrt(self.map_cbf) / self.ref_pressure)
        
    def get_cbf_map(self, ws: torch.Tensor = None, dvecs: torch.Tensor = None) -> torch.Tensor:
        """Return CBF result map
        
        Args:
            ws: Optional custom steering vectors
            dvecs: Optional custom delay vectors
            
        Returns:
            torch.Tensor: CBF result map
        """
        self._compute_cbf_map(ws, dvecs)
        return self.map_cbf
        
    def get_cbf_map_db(self, ws: torch.Tensor = None, dvecs: torch.Tensor = None) -> torch.Tensor:
        """Return CBF result map (dB)
        
        Args:
            ws: Optional custom steering vectors
            dvecs: Optional custom delay vectors
            
        Returns:
            torch.Tensor: CBF result map (dB)
        """
        self._compute_cbf_map(ws, dvecs)
        return self.map_cbf_db
        
    def get_scan_points(self):
        """Return scan point coordinates"""
        return self.scan_points
        
    def get_steering_vectors(self):
        """Return steering vectors"""
        self._compute_steering_vectors()
        return self.ws, self.dvecs
    
    def cal_special_dvecs(self, x_idx, y_idx):
        """
        Calculate steering vectors for specified grid position
        
        Args:
            x_idx: Grid index in x direction
            y_idx: Grid index in y direction
        Returns:
            w: Steering vector at this position
            dvec: Delay vector at this position
        """
        scan_idx = x_idx * self.x_grid + y_idx
        x, y, z = self.scan_points[scan_idx]
        
        # Calculate steering vector for a single position
        source_pos = torch.tensor([x, y, z], device=self.device).unsqueeze(0)  # (1, 3)
        distances = torch.sqrt(torch.sum((source_pos - self.array_positions)**2, dim=1))
        
        time_delays = distances / self.c
        dvec = torch.exp(-2j * torch.pi * self.act_freq * time_delays)
        
        if self.if_ref_distance_1:
            ref_distance = 1.0
        else:
            ref_distance = torch.sqrt(torch.sum(source_pos**2))
        dvec_consider_r = ref_distance * (1.0/distances) * dvec
        w_consider_r = distances * (1.0/ref_distance) * dvec
        
        return w_consider_r, dvec_consider_r
        
    def get_spatial_correlation_matrix(self):
        """Return spatial correlation matrix"""
        self._compute_scm()
        return self.SCM
        
    def get_stft_data(self):
        """Return STFT data"""
        self._compute_stft()
        return self.stft_data
    
    def get_act_bin_freq(self):
        """Return actual frequency bin and corresponding frequency"""
        return self.act_bin, self.act_freq
    
def get_steering_vector(x: float, y: float, z: float, freq: float, c: float, mic_positions: np.ndarray, 
                        coordinate_type: str = 'cartesian', if_torch: bool = False, if_ref_distance_1: bool = False) -> tuple:
    """Calculate steering vector for given source position and microphone array.
    
    Args:
        x: x coordinate or azimuth angle
        y: y coordinate or elevation angle 
        z: z coordinate or distance
        freq: Frequency in Hz
        c: Speed of sound in m/s
        mic_positions: Array of shape (n_mics, 3) containing microphone positions
        coordinate_type: 'cartesian' or 'spherical'
        if_torch: Whether to use PyTorch tensors
        
    Returns:
        w_consider_r: Steering vector considering distance attenuation
        dvec_consider_r: Direction vector considering distance attenuation
    """
    # Convert spherical to cartesian if needed
    if coordinate_type == 'spherical':
        # Convert from spherical (azimuth, elevation, distance) to cartesian (x,y,z)
        azimuth, elevation, r = x, y, z
        if if_torch:
            x = r * torch.cos(elevation) * torch.cos(azimuth)
            y = r * torch.cos(elevation) * torch.sin(azimuth)
            z = r * torch.sin(elevation)
        else:
            x = r * np.cos(elevation) * np.cos(azimuth)
            y = r * np.cos(elevation) * np.sin(azimuth) 
            z = r * np.sin(elevation)
    
    if if_torch:
        source_pos = torch.tensor([x, y, z], device=mic_positions.device)
        # Calculate distances from source to each microphone
        distances = torch.sqrt(torch.sum((mic_positions - source_pos)**2, dim=1))
        # Calculate time delays
        time_delays = distances / c
        # Calculate basic steering vector
        dvec = torch.exp(-2j * torch.pi * freq * time_delays)
    else:
        source_pos = np.array([x, y, z])
        # Calculate distances from source to each microphone
        distances = np.sqrt(np.sum((mic_positions - source_pos)**2, axis=1))
        # Calculate time delays
        time_delays = distances / c
        # Calculate basic steering vector
        dvec = np.exp(-2j * np.pi * freq * time_delays)
    
    # Calculate reference distance (distance to origin)
    if if_ref_distance_1:
        ref_distance = 1
    else:
        ref_distance = torch.sqrt(torch.sum(source_pos**2)) if if_torch else np.sqrt(np.sum(source_pos**2))

    # Calculate steering vectors with distance compensation
    dvec_consider_r = ref_distance * (1.0/distances) * dvec
    w_consider_r = distances * (1.0/ref_distance) * dvec
    
    return w_consider_r.to(torch.complex64), dvec_consider_r.to(torch.complex64) 