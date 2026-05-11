import os
import time
import threading
import queue
import struct
import numpy as np
import scipy.io.wavfile as wav
import scipy.fftpack as fftpack
from scipy.signal import decimate, find_peaks
import sounddevice as sd
import psutil
import logging
from enum import Enum
from collections import Counter, deque

# Handle PySerial import with fallback
try:
    import serial
    if not hasattr(serial, 'Serial'):
        raise ImportError("PySerial Serial class not found")
    SERIAL_AVAILABLE = True
    print("✓ PySerial loaded successfully")
except ImportError as e:
    print(f"⚠️  PySerial not available: {e}")
    print("   Install with: pip install pyserial")
    SERIAL_AVAILABLE = False
    serial = None

# Aircraft Engine Configuration
AIRCRAFT_TYPE = "PISTON"  # Options: "PISTON", "TURBOPROP", "TURBOFAN"
ENGINE_MODEL = "LYCOMING_IO360"  # Specific engine model

# Audio Settings
RECORD_SECONDS = 5
SAMPLE_RATE = 44100
AUDIO_DIR = "./recordings"
DECIMATED_RATE = 1000  # Higher for aircraft (was 500)
CLEAN_INTERVAL_DAYS = 7
DISK_SPACE_THRESHOLD = 0.1
MIN_FREE_SPACE_GB = 1.0
ANALYZE_QUEUE_SIZE = 10
MIC_DEVICE = None
LOG_FILE = "aircraft_audio_system.log"

# UART Settings
UART_PORT = "/dev/serial0"
UART_BAUDRATE = 230400
ACK_TIMEOUT = 1.0
MAX_RETRIES = 5
INTER_PACKET_DELAY = 0.05

# UART Protocol Constants
START_BYTE_DATA = 0xAA
START_BYTE_RECORD = 0xAC
START_BYTE_SYNC = 0xAB
START_BYTE_ALERT = 0xAD  # New: Aircraft alerts
END_BYTE = 0x55

# Aircraft-Specific Constants
RPM_CONFIDENCE_THRESHOLD = 0.6
STATE_CHANGE_MIN_DURATION = 3.0  # Minimum seconds before state change
OVERSPEED_ALERT_THRESHOLD = 0.95  # 95% of redline

# Setup enhanced logging for aircraft
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(threadName)s] - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()  # Also log to console for aircraft operations
    ]
)

os.makedirs(AUDIO_DIR, exist_ok=True)

# Thread-safe queues
analyze_queue = queue.Queue(maxsize=ANALYZE_QUEUE_SIZE)
alert_queue = queue.Queue(maxsize=20)

# UART variables
ser = None
uart_lock = threading.Lock()
file_counter = 1
last_cleanup = time.time()
packet_seq_num = 0

# Aircraft Engine Types and States
class AircraftEngineType(Enum):
    PISTON = "piston"
    TURBOPROP = "turboprop"
    TURBOFAN = "turbofan"

class AircraftEngineState(Enum):
    SHUTDOWN = 0
    STARTING = 1
    GROUND_IDLE = 2
    TAXI_POWER = 3
    RUNUP = 4
    TAKEOFF_POWER = 5
    CLIMB_POWER = 6
    CRUISE_POWER = 7
    DESCENT_POWER = 8
    APPROACH_POWER = 9
    OVERSPEED = 10
    ABNORMAL = 11

class AlertLevel(Enum):
    INFO = 1
    WARNING = 2
    CAUTION = 3
    CRITICAL = 4

# Enhanced Datum Keys for Aircraft
class RawDatumKey:
    AUDIO_ARRAY = "audio_array"
    SAMPLE_RATE = "sample_rate"

class DerivedDataKey:
    DECIMATED_AUDIO = "decimated_audio"
    DECIMATED_AUDIO_RATE = "decimated_audio_rate"
    FREQUENCY_PEAK = "frequency_peak"
    FREQUENCY_CONFIDENCE = "frequency_confidence"
    FREQUENCY_PEAKS = "frequency_peaks"  # Multiple peaks
    HARMONIC_CONTENT = "harmonic_content"
    FFT_DATA = "fft_data"  # FFT magnitude spectrum
    FFT_FREQUENCIES = "fft_frequencies"  # Corresponding frequencies
    FFT_METADATA = "fft_metadata"  # FFT analysis metadata
    SPECTRAL_ANALYSIS = "spectral_analysis"  # Detailed spectral features
    RPM = "rpm"
    RPM_PERCENT = "rpm_percent"
    RPM_CONFIDENCE = "rpm_confidence"
    RPM_TREND = "rpm_trend"
    ENGINE_STATE = "engine_state"
    ENGINE_STATE_NAME = "engine_state_name"
    ENGINE_HEALTH_SCORE = "engine_health_score"
    VIBRATION_SIGNATURE = "vibration_signature"

class Datum:
    def __init__(self, audio_array, sample_rate):
        self._raw_data = {
            RawDatumKey.AUDIO_ARRAY: audio_array, 
            RawDatumKey.SAMPLE_RATE: sample_rate
        }
        self._derived_data = {}
        self.timestamp = time.time()

    def get_raw_datum(self, key):
        return self._raw_data.get(key)

    def set_raw_datum(self, key, value):
        self._raw_data[key] = value

    def get_derived_data(self, key):
        return self._derived_data.get(key)

    def set_derived_data(self, key, value):
        self._derived_data[key] = value

# Aircraft Engine Configurations
class AircraftEngineConfig:
    """Base configuration for aircraft engines"""
    def __init__(self):
        self.engine_type = None
        self.model_name = ""
        self.events_per_cycle = 2
        
        # Frequency analysis ranges
        self.freq_range_min = 20
        self.freq_range_max = 300
        
        # RPM ranges
        self.shutdown_max = 100
        self.idle_min = 600
        self.idle_max = 900
        self.cruise_min = 2000
        self.cruise_max = 2500
        self.redline = 2700
        
        # State thresholds with hysteresis
        self.state_thresholds = {}

class PistonEngineConfig(AircraftEngineConfig):
    """Configuration for piston aircraft engines"""
    def __init__(self, model="LYCOMING_IO360"):
        super().__init__()
        self.engine_type = AircraftEngineType.PISTON
        self.model_name = model
        self.events_per_cycle = 2  # 4-stroke engine
        
        # Frequency ranges for piston engines
        self.freq_range_min = 15
        self.freq_range_max = 200
        
        if model == "LYCOMING_IO360":
            self.idle_min = 650
            self.idle_max = 850
            self.cruise_min = 2000
            self.cruise_max = 2500
            self.redline = 2700
        elif model == "CONTINENTAL_IO550":
            self.idle_min = 600
            self.idle_max = 800
            self.cruise_min = 2300
            self.cruise_max = 2700
            self.redline = 2850
        
        # State thresholds with hysteresis
        self.state_thresholds = {
            AircraftEngineState.STARTING: (200, 600),
            AircraftEngineState.GROUND_IDLE: (self.idle_min, self.idle_max),
            AircraftEngineState.TAXI_POWER: (900, 1200),
            AircraftEngineState.RUNUP: (1400, 1800),
            AircraftEngineState.TAKEOFF_POWER: (2300, self.redline),
            AircraftEngineState.CRUISE_POWER: (self.cruise_min, self.cruise_max),
        }

class TurbopropEngineConfig(AircraftEngineConfig):
    """Configuration for turboprop engines"""
    def __init__(self, model="PT6A"):
        super().__init__()
        self.engine_type = AircraftEngineType.TURBOPROP
        self.model_name = model
        self.events_per_cycle = 1  # Continuous combustion
        
        # Turboprops use % RPM
        self.use_percent_rpm = True
        self.freq_range_min = 25
        self.freq_range_max = 300
        
        # Percent RPM thresholds
        self.ground_idle_min = 50
        self.ground_idle_max = 65
        self.flight_idle_min = 65
        self.flight_idle_max = 75
        self.cruise_min = 80
        self.cruise_max = 95
        self.redline = 100

# Feature Extraction Interface
class FeatureExtraction:
    def extract_features(self, datum: Datum) -> Datum:
        raise NotImplementedError

# Enhanced Decimation for Aircraft
class AircraftDecimation(FeatureExtraction):
    def __init__(self, target_rate=1000):
        self.target_rate = target_rate

    def extract_features(self, datum: Datum):
        signal = datum.get_raw_datum(RawDatumKey.AUDIO_ARRAY)
        original_rate = datum.get_raw_datum(RawDatumKey.SAMPLE_RATE)
        
        try:
            decimation_factor = int(original_rate // self.target_rate)
            if decimation_factor < 1:
                decimated_signal = signal
                decimated_rate = original_rate
            else:
                # Use anti-aliasing filter for aircraft precision
                decimated_signal = decimate(signal, decimation_factor, 
                                          axis=0, ftype='iir', zero_phase=True)
                decimated_rate = original_rate // decimation_factor
            
            datum.set_derived_data(DerivedDataKey.DECIMATED_AUDIO, decimated_signal)
            datum.set_derived_data(DerivedDataKey.DECIMATED_AUDIO_RATE, decimated_rate)
            
            logging.debug(f"Decimated from {original_rate}Hz to {decimated_rate}Hz")
            
        except Exception as e:
            logging.error(f"Aircraft decimation error: {e}")
            datum.set_derived_data(DerivedDataKey.DECIMATED_AUDIO, signal)
            datum.set_derived_data(DerivedDataKey.DECIMATED_AUDIO_RATE, original_rate)
        
        return datum

# Enhanced Frequency Analysis for Aircraft
class AircraftFrequencyAnalyzer(FeatureExtraction):
    def __init__(self, engine_config):
        self.config = engine_config
        self.freq_history = deque(maxlen=10)  # Track frequency stability

    def extract_features(self, datum: Datum):
        try:
            # Process both original and decimated signals
            self._analyze_signal(datum, use_decimated=False)
            self._analyze_signal(datum, use_decimated=True)
            
            # Calculate frequency stability
            freq = datum.get_derived_data(DerivedDataKey.FREQUENCY_PEAK) or 0.0
            self.freq_history.append(freq)
            
            if len(self.freq_history) >= 3:
                freq_std = np.std(self.freq_history)
                freq_mean = np.mean(self.freq_history)
                stability = 1.0 - min(1.0, freq_std / (freq_mean + 1e-10))
            else:
                stability = 0.5
            
            # Enhanced confidence calculation
            confidence = self._calculate_enhanced_confidence(datum, stability)
            datum.set_derived_data(DerivedDataKey.FREQUENCY_CONFIDENCE, confidence)
            
        except Exception as e:
            logging.error(f"Aircraft frequency analysis error: {e}")
            self._set_default_values(datum)
        
        return datum

    def _analyze_signal(self, datum: Datum, use_decimated=True):
        """Analyze signal with aircraft-specific enhancements"""
        
        if use_decimated:
            signal = datum.get_derived_data(DerivedDataKey.DECIMATED_AUDIO)
            sample_rate = datum.get_derived_data(DerivedDataKey.DECIMATED_AUDIO_RATE)
            key_suffix = "_DECIMATED"
        else:
            signal = datum.get_raw_datum(RawDatumKey.AUDIO_ARRAY)
            sample_rate = datum.get_raw_datum(RawDatumKey.SAMPLE_RATE)
            key_suffix = ""

        if signal is None or len(signal) == 0:
            return

        # Enhanced windowing for aircraft
        window = np.hanning(len(signal))
        windowed_signal = signal * window

        # Zero-padding for better frequency resolution
        next_power_2 = 2 ** int(np.ceil(np.log2(len(signal))))
        if next_power_2 > len(signal):
            padded_signal = np.pad(windowed_signal, 
                                 (0, next_power_2 - len(signal)), 
                                 mode='constant')
        else:
            padded_signal = windowed_signal

        # Compute FFT
        N = len(padded_signal)
        fft_complex = fftpack.fft(padded_signal)
        fft_data = np.abs(fft_complex[:N//2])
        freqs = fftpack.fftfreq(N, 1/sample_rate)[:N//2]

        # Store complete FFT data for export (decimated signal only)
        if use_decimated:
            # Create FFT metadata
            fft_metadata = {
                'sample_rate': sample_rate,
                'window_type': 'hanning',
                'fft_size': N,
                'frequency_resolution': sample_rate / N,
                'analysis_range': (self.config.freq_range_min, self.config.freq_range_max),
                'timestamp': time.time(),
                'zero_padded': next_power_2 > len(signal),
                'original_length': len(signal),
                'padded_length': N
            }
            
            # Store full FFT data and frequencies
            datum.set_derived_data(DerivedDataKey.FFT_DATA, fft_data.tolist())
            datum.set_derived_data(DerivedDataKey.FFT_FREQUENCIES, freqs.tolist())
            datum.set_derived_data(DerivedDataKey.FFT_METADATA, fft_metadata)
            
            # Calculate additional spectral features
            spectral_features = self._calculate_spectral_features(fft_data, freqs)
            datum.set_derived_data(DerivedDataKey.SPECTRAL_ANALYSIS, spectral_features)

        # Aircraft-specific frequency range
        valid_range = ((freqs >= self.config.freq_range_min) & 
                      (freqs <= self.config.freq_range_max))

        if np.any(valid_range):
            valid_fft = fft_data[valid_range]
            valid_freqs = freqs[valid_range]
            
            # Find multiple peaks for aircraft engines
            peaks = self._find_aircraft_peaks(valid_fft, valid_freqs)
            
            # Select best engine frequency
            engine_freq = self._select_engine_frequency(peaks, valid_fft, valid_freqs)
            
            # Calculate harmonic content
            harmonic_strength = self._calculate_harmonic_content(peaks, engine_freq)
            
            if use_decimated:
                datum.set_derived_data(DerivedDataKey.FREQUENCY_PEAK, engine_freq)
                datum.set_derived_data(DerivedDataKey.FREQUENCY_PEAKS, peaks)
                datum.set_derived_data(DerivedDataKey.HARMONIC_CONTENT, harmonic_strength)
        else:
            if use_decimated:
                datum.set_derived_data(DerivedDataKey.FREQUENCY_PEAK, 0.0)
                datum.set_derived_data(DerivedDataKey.FREQUENCY_PEAKS, [])
                datum.set_derived_data(DerivedDataKey.HARMONIC_CONTENT, 0.0)

    def _calculate_spectral_features(self, fft_data, freqs):
        """Calculate additional spectral analysis features"""
        
        try:
            # Spectral centroid (weighted mean frequency)
            if np.sum(fft_data) > 0:
                spectral_centroid = np.sum(freqs * fft_data) / np.sum(fft_data)
            else:
                spectral_centroid = 0.0
            
            # Spectral spread (frequency distribution width)
            if np.sum(fft_data) > 0:
                spectral_spread = np.sqrt(np.sum(((freqs - spectral_centroid) ** 2) * fft_data) / np.sum(fft_data))
            else:
                spectral_spread = 0.0
            
            # Spectral rolloff (95% of energy below this frequency)
            cumulative_energy = np.cumsum(fft_data ** 2)
            total_energy = cumulative_energy[-1]
            rolloff_idx = np.where(cumulative_energy >= 0.95 * total_energy)[0]
            spectral_rolloff = freqs[rolloff_idx[0]] if len(rolloff_idx) > 0 else freqs[-1]
            
            # Spectral flux (change from previous analysis)
            if hasattr(self, 'previous_fft'):
                # Resample to match if needed
                if len(self.previous_fft) == len(fft_data):
                    spectral_flux = np.sum((fft_data - self.previous_fft) ** 2)
                else:
                    spectral_flux = 0.0
            else:
                spectral_flux = 0.0
            
            self.previous_fft = fft_data.copy()
            
            # Peak-to-average ratio in analysis band
            analysis_range = ((freqs >= self.config.freq_range_min) & 
                            (freqs <= self.config.freq_range_max))
            if np.any(analysis_range):
                analysis_fft = fft_data[analysis_range]
                peak_magnitude = np.max(analysis_fft)
                avg_magnitude = np.mean(analysis_fft)
                peak_to_avg = peak_magnitude / (avg_magnitude + 1e-10)
            else:
                peak_to_avg = 0.0
            
            # Signal-to-noise ratio estimate
            if np.any(analysis_range):
                # Estimate noise floor as 10th percentile
                noise_floor = np.percentile(analysis_fft, 10)
                signal_level = np.max(analysis_fft)
                snr_estimate = 20 * np.log10((signal_level + 1e-10) / (noise_floor + 1e-10))
            else:
                snr_estimate = 0.0
            
            # Fundamental frequency candidates (top 5 peaks in analysis range)
            fundamental_candidates = []
            if np.any(analysis_range):
                analysis_freqs = freqs[analysis_range]
                peaks, properties = find_peaks(analysis_fft, 
                                             prominence=np.max(analysis_fft) * 0.1,
                                             distance=5)
                if len(peaks) > 0:
                    peak_freqs = analysis_freqs[peaks]
                    peak_mags = analysis_fft[peaks]
                    # Sort by magnitude and take top 5
                    sorted_indices = np.argsort(peak_mags)[::-1]
                    for i in sorted_indices[:5]:
                        fundamental_candidates.append({
                            'frequency': float(peak_freqs[i]),
                            'magnitude': float(peak_mags[i]),
                            'prominence': float(properties['prominences'][i]) if 'prominences' in properties else 0.0
                        })
            
            return {
                'spectral_centroid': float(spectral_centroid),
                'spectral_spread': float(spectral_spread),
                'spectral_rolloff': float(spectral_rolloff),
                'spectral_flux': float(spectral_flux),
                'peak_to_avg_ratio': float(peak_to_avg),
                'snr_estimate': float(snr_estimate),
                'fundamental_candidates': fundamental_candidates,
                'total_energy': float(np.sum(fft_data ** 2)),
                'analysis_band_energy': float(np.sum(fft_data[analysis_range] ** 2)) if np.any(analysis_range) else 0.0
            }
            
        except Exception as e:
            logging.error(f"Spectral features calculation error: {e}")
            return {
                'spectral_centroid': 0.0,
                'spectral_spread': 0.0,
                'spectral_rolloff': 0.0,
                'spectral_flux': 0.0,
                'peak_to_avg_ratio': 0.0,
                'snr_estimate': 0.0,
                'fundamental_candidates': [],
                'total_energy': 0.0,
                'analysis_band_energy': 0.0
            }

    def _find_aircraft_peaks(self, fft_data, freqs):
        """Find multiple frequency peaks suitable for aircraft engines"""
        
        # Use scipy's find_peaks with aircraft-appropriate parameters
        prominence = np.max(fft_data) * 0.15  # 15% prominence threshold
        peaks, properties = find_peaks(fft_data, 
                                     prominence=prominence,
                                     distance=int(len(fft_data) * 0.02))  # 2% min distance
        
        peak_freqs = freqs[peaks]
        peak_magnitudes = fft_data[peaks]
        
        # Sort by magnitude
        peak_data = list(zip(peak_freqs, peak_magnitudes))
        peak_data.sort(key=lambda x: x[1], reverse=True)
        
        return peak_data[:10]  # Return top 10 peaks

    def _select_engine_frequency(self, peaks, fft_data, freqs):
        """Select most likely engine frequency from peaks"""
        
        if not peaks:
            return 0.0
        
        # For aircraft, prefer frequencies with strong harmonic content
        best_freq = 0.0
        best_score = 0.0
        
        for freq, magnitude in peaks[:5]:  # Check top 5 peaks
            # Score based on magnitude and harmonic content
            harmonic_score = self._score_harmonics(freq, peaks)
            total_score = magnitude * (1.0 + harmonic_score)
            
            if total_score > best_score:
                best_score = total_score
                best_freq = freq
        
        return best_freq

    def _score_harmonics(self, fundamental, peaks):
        """Score based on harmonic content"""
        
        if fundamental == 0:
            return 0.0
        
        harmonic_score = 0.0
        for harmonic_num in [2, 3, 4]:
            target_freq = fundamental * harmonic_num
            
            # Find closest peak to harmonic frequency
            for freq, magnitude in peaks:
                if abs(freq - target_freq) < (fundamental * 0.1):  # Within 10%
                    harmonic_score += 0.3  # Each harmonic adds to score
                    break
        
        return harmonic_score

    def _calculate_harmonic_content(self, peaks, fundamental):
        """Calculate overall harmonic strength"""
        
        if not peaks or fundamental == 0:
            return 0.0
        
        fundamental_magnitude = 0.0
        harmonic_magnitude = 0.0
        
        # Find fundamental magnitude
        for freq, magnitude in peaks:
            if abs(freq - fundamental) < (fundamental * 0.05):  # Within 5%
                fundamental_magnitude = magnitude
                break
        
        # Sum harmonic magnitudes
        for harmonic_num in [2, 3, 4, 5]:
            target_freq = fundamental * harmonic_num
            for freq, magnitude in peaks:
                if abs(freq - target_freq) < (fundamental * 0.1):
                    harmonic_magnitude += magnitude
                    break
        
        if fundamental_magnitude > 0:
            return harmonic_magnitude / fundamental_magnitude
        return 0.0

    def _calculate_enhanced_confidence(self, datum, frequency_stability):
        """Calculate enhanced confidence for aircraft applications"""
        
        freq = datum.get_derived_data(DerivedDataKey.FREQUENCY_PEAK) or 0.0
        harmonic_content = datum.get_derived_data(DerivedDataKey.HARMONIC_CONTENT) or 0.0
        peaks = datum.get_derived_data(DerivedDataKey.FREQUENCY_PEAKS) or []
        
        if freq == 0 or not peaks:
            return 0.0
        
        # Multiple confidence factors
        confidence_factors = []
        
        # 1. Frequency stability over time
        confidence_factors.append(frequency_stability)
        
        # 2. Harmonic content (engines have harmonics)
        harmonic_confidence = min(1.0, harmonic_content)
        confidence_factors.append(harmonic_confidence)
        
        # 3. Peak prominence
        if len(peaks) >= 2:
            primary_mag = peaks[0][1]
            secondary_mag = peaks[1][1]
            prominence = primary_mag / (secondary_mag + 1e-10)
            prominence_confidence = min(1.0, prominence / 5.0)  # Normalize
            confidence_factors.append(prominence_confidence)
        
        # 4. Frequency range appropriateness
        range_confidence = 1.0
        if freq < self.config.freq_range_min * 1.2 or freq > self.config.freq_range_max * 0.8:
            range_confidence = 0.5  # Lower confidence at range edges
        confidence_factors.append(range_confidence)
        
        # Combine confidence factors (geometric mean for conservative estimate)
        if confidence_factors:
            combined_confidence = np.power(np.prod(confidence_factors), 1.0/len(confidence_factors))
            return float(combined_confidence)
        
        return 0.5

    def _set_default_values(self, datum):
        """Set default values on error"""
        datum.set_derived_data(DerivedDataKey.FREQUENCY_PEAK, 0.0)
        datum.set_derived_data(DerivedDataKey.FREQUENCY_CONFIDENCE, 0.0)
        datum.set_derived_data(DerivedDataKey.FREQUENCY_PEAKS, [])
        datum.set_derived_data(DerivedDataKey.HARMONIC_CONTENT, 0.0)

# Aircraft RPM Calculator
class AircraftRPMCalculator(FeatureExtraction):
    def __init__(self, engine_config):
        self.config = engine_config
        self.rpm_history = deque(maxlen=15)  # 15 measurements for trend analysis

    def extract_features(self, datum: Datum):
        try:
            peak_freq = datum.get_derived_data(DerivedDataKey.FREQUENCY_PEAK) or 0.0
            confidence = datum.get_derived_data(DerivedDataKey.FREQUENCY_CONFIDENCE) or 0.0
            
            if peak_freq <= 0 or confidence < 0.2:
                rpm = 0.0
                rpm_percent = 0.0
                rpm_confidence = 0.0
            else:
                # Calculate RPM using aircraft-specific formula
                rpm = (peak_freq * 60) / self.config.events_per_cycle
                
                # Validate RPM range for aircraft
                if self._validate_aircraft_rpm(rpm):
                    # Calculate percent RPM for turbine engines
                    if hasattr(self.config, 'use_percent_rpm'):
                        rpm_percent = (rpm / 1900) * 100  # Typical turboprop prop RPM
                    else:
                        rpm_percent = (rpm / self.config.redline) * 100
                    
                    rpm_confidence = confidence
                else:
                    rpm = 0.0
                    rpm_percent = 0.0
                    rpm_confidence = 0.0
                    logging.warning(f"Invalid aircraft RPM: {rpm:.1f} from freq {peak_freq:.1f}Hz")
            
            # Update RPM history and calculate trend
            self.rpm_history.append(rpm)
            rpm_trend = self._calculate_rpm_trend()
            
            # Store results
            datum.set_derived_data(DerivedDataKey.RPM, rpm)
            datum.set_derived_data(DerivedDataKey.RPM_PERCENT, rpm_percent)
            datum.set_derived_data(DerivedDataKey.RPM_CONFIDENCE, rpm_confidence)
            datum.set_derived_data(DerivedDataKey.RPM_TREND, rpm_trend)
            
            logging.debug(f"Aircraft RPM: {rpm:.1f} ({rpm_percent:.1f}%), "
                         f"Confidence: {rpm_confidence:.2f}, Trend: {rpm_trend}")
            
        except Exception as e:
            logging.error(f"Aircraft RPM calculation error: {e}")
            self._set_default_rpm_values(datum)
        
        return datum

    def _validate_aircraft_rpm(self, rpm):
        """Validate RPM for aircraft engines"""
        
        if self.config.engine_type == AircraftEngineType.PISTON:
            return 0 <= rpm <= (self.config.redline * 1.05)  # Allow 5% over redline
        elif self.config.engine_type == AircraftEngineType.TURBOPROP:
            # For turboprops, validate prop RPM (typically 0-2000)
            return 0 <= rpm <= 2200
        else:
            # For turbofans, would need specific validation
            return 0 <= rpm <= 20000
    
    def _calculate_rpm_trend(self):
        """Calculate RPM trend over recent measurements"""
        
        if len(self.rpm_history) < 5:
            return "stable"
        
        recent_rpms = list(self.rpm_history)
        
        # Calculate linear trend
        x = np.arange(len(recent_rpms))
        coeffs = np.polyfit(x, recent_rpms, 1)
        slope = coeffs[0]
        
        # Classify trend
        if slope > 50:  # Increasing more than 50 RPM per measurement
            return "increasing"
        elif slope < -50:  # Decreasing more than 50 RPM per measurement
            return "decreasing"
        else:
            return "stable"
    
    def _set_default_rpm_values(self, datum):
        """Set default RPM values on error"""
        datum.set_derived_data(DerivedDataKey.RPM, 0.0)
        datum.set_derived_data(DerivedDataKey.RPM_PERCENT, 0.0)
        datum.set_derived_data(DerivedDataKey.RPM_CONFIDENCE, 0.0)
        datum.set_derived_data(DerivedDataKey.RPM_TREND, "unknown")

# Aircraft Engine State Detector
class AircraftEngineStateDetector(FeatureExtraction):
    def __init__(self, engine_config):
        self.config = engine_config
        self.current_state = AircraftEngineState.SHUTDOWN
        self.state_history = deque(maxlen=10)
        self.state_change_time = time.time()
        self.last_rpm_values = deque(maxlen=20)

    def extract_features(self, datum: Datum):
        try:
            rpm = datum.get_derived_data(DerivedDataKey.RPM) or 0.0
            rpm_confidence = datum.get_derived_data(DerivedDataKey.RPM_CONFIDENCE) or 0.0
            rpm_trend = datum.get_derived_data(DerivedDataKey.RPM_TREND) or "unknown"
            
            # Classify state based on engine type
            if self.config.engine_type == AircraftEngineType.PISTON:
                new_state = self._classify_piston_state(rpm, rpm_confidence, rpm_trend)
            else:
                rpm_percent = datum.get_derived_data(DerivedDataKey.RPM_PERCENT) or 0.0
                new_state = self._classify_turbine_state(rpm_percent, rpm_confidence, rpm_trend)
            
            # Apply temporal filtering and hysteresis
            filtered_state = self._apply_state_filtering(new_state, rpm, rpm_confidence)
            
            # Check for abnormal conditions
            self._check_aircraft_abnormal_conditions(datum, filtered_state)
            
            # Calculate engine health score
            health_score = self._calculate_engine_health(datum, filtered_state)
            
            # Store results
            datum.set_derived_data(DerivedDataKey.ENGINE_STATE, filtered_state.value)
            datum.set_derived_data(DerivedDataKey.ENGINE_STATE_NAME, filtered_state.name)
            datum.set_derived_data(DerivedDataKey.ENGINE_HEALTH_SCORE, health_score)
            
            # Log state changes
            if filtered_state != self.current_state:
                logging.info(f"Aircraft engine state change: {self.current_state.name} → "
                           f"{filtered_state.name} (RPM: {rpm:.1f})")
                self.current_state = filtered_state
                self.state_change_time = time.time()
            
        except Exception as e:
            logging.error(f"Aircraft engine state detection error: {e}")
            datum.set_derived_data(DerivedDataKey.ENGINE_STATE, AircraftEngineState.ABNORMAL.value)
            datum.set_derived_data(DerivedDataKey.ENGINE_STATE_NAME, "ABNORMAL")
            datum.set_derived_data(DerivedDataKey.ENGINE_HEALTH_SCORE, 0.0)
        
        return datum

    def _classify_piston_state(self, rpm, confidence, trend):
        """Classify piston aircraft engine state"""
        
        if confidence < RPM_CONFIDENCE_THRESHOLD or rpm < self.config.shutdown_max:
            return AircraftEngineState.SHUTDOWN
        
        elif 200 <= rpm < 600:
            return AircraftEngineState.STARTING
        
        elif self.config.idle_min <= rpm <= self.config.idle_max:
            return AircraftEngineState.GROUND_IDLE
        
        elif 900 <= rpm < 1200:
            return AircraftEngineState.TAXI_POWER
        
        elif 1400 <= rpm < 1800:
            # Could be runup or approach power
            if trend == "increasing":
                return AircraftEngineState.RUNUP
            else:
                return AircraftEngineState.APPROACH_POWER
        
        elif 1800 <= rpm < 2200:
            return AircraftEngineState.DESCENT_POWER
        
        elif 2200 <= rpm < 2500:
            return AircraftEngineState.CRUISE_POWER
        
        elif 2500 <= rpm <= self.config.redline:
            return AircraftEngineState.TAKEOFF_POWER
        
        elif rpm > self.config.redline:
            return AircraftEngineState.OVERSPEED
        
        else:
            return AircraftEngineState.ABNORMAL

    def _classify_turbine_state(self, rpm_percent, confidence, trend):
        """Classify turbine aircraft engine state"""
        
        if confidence < RPM_CONFIDENCE_THRESHOLD or rpm_percent < 15:
            return AircraftEngineState.SHUTDOWN
        
        elif 15 <= rpm_percent < 25:
            return AircraftEngineState.STARTING
        
        elif 25 <= rpm_percent < 35:
            return AircraftEngineState.GROUND_IDLE
        
        elif 35 <= rpm_percent < 50:
            return AircraftEngineState.TAXI_POWER
        
        elif 50 <= rpm_percent < 70:
            return AircraftEngineState.APPROACH_POWER
        
        elif 70 <= rpm_percent < 85:
            return AircraftEngineState.CRUISE_POWER
        
        elif 85 <= rpm_percent <= 100:
            return AircraftEngineState.TAKEOFF_POWER
        
        elif rpm_percent > 100:
            return AircraftEngineState.OVERSPEED
        
        else:
            return AircraftEngineState.ABNORMAL

    def _apply_state_filtering(self, new_state, rpm, confidence):
        """Apply temporal filtering and hysteresis"""
        
        # Add to history
        self.state_history.append(new_state)
        self.last_rpm_values.append(rpm)
        
        # Require minimum time in state before allowing change
        time_in_current_state = time.time() - self.state_change_time
        if time_in_current_state < STATE_CHANGE_MIN_DURATION:
            # Don't change state too quickly unless it's critical
            if new_state not in [AircraftEngineState.OVERSPEED, AircraftEngineState.ABNORMAL]:
                return self.current_state
        
        # Require consensus from recent measurements
        if len(self.state_history) >= 3:
            state_counts = Counter(self.state_history[-5:])  # Last 5 measurements
            most_common_state, count = state_counts.most_common(1)[0]
            
            # Require 60% consensus for state change
            if count / len(self.state_history[-5:]) >= 0.6:
                return most_common_state
        
        return self.current_state

    def _check_aircraft_abnormal_conditions(self, datum, state):
        """Check for aircraft-specific abnormal conditions"""
        
        rpm = datum.get_derived_data(DerivedDataKey.RPM) or 0.0
        rpm_percent = datum.get_derived_data(DerivedDataKey.RPM_PERCENT) or 0.0
        rpm_trend = datum.get_derived_data(DerivedDataKey.RPM_TREND) or "unknown"
        confidence = datum.get_derived_data(DerivedDataKey.RPM_CONFIDENCE) or 0.0
        
        alerts = []
        
        # Overspeed detection
        overspeed_threshold = self.config.redline * OVERSPEED_ALERT_THRESHOLD
        if ((self.config.engine_type == AircraftEngineType.PISTON and rpm > overspeed_threshold) or
            (self.config.engine_type != AircraftEngineType.PISTON and rpm_percent > 95)):
            alerts.append((AlertLevel.CRITICAL, f"ENGINE OVERSPEED: {rpm:.1f} RPM ({rpm_percent:.1f}%)"))
        
        # Rapid RPM changes
        if len(self.last_rpm_values) >= 5:
            recent_change = abs(self.last_rpm_values[-1] - self.last_rpm_values[-5])
            if recent_change > 300:  # More than 300 RPM change in 25 seconds
                alerts.append((AlertLevel.WARNING, f"Rapid RPM change: {recent_change:.1f} RPM/25sec"))
        
        # Low confidence readings during operation
        if state != AircraftEngineState.SHUTDOWN and confidence < 0.3:
            alerts.append((AlertLevel.CAUTION, f"Low RPM confidence: {confidence:.2f}"))
        
        # Frequency analysis issues
        harmonic_content = datum.get_derived_data(DerivedDataKey.HARMONIC_CONTENT) or 0.0
        if state != AircraftEngineState.SHUTDOWN and harmonic_content < 0.1:
            alerts.append((AlertLevel.CAUTION, "Poor harmonic signature - possible engine irregularity"))
        
        # Engine state inconsistencies
        if state == AircraftEngineState.ABNORMAL:
            alerts.append((AlertLevel.WARNING, f"Engine state classification failed at {rpm:.1f} RPM"))
        
        # Queue alerts for transmission
        for alert_level, message in alerts:
            try:
                alert_queue.put_nowait((time.time(), alert_level, message))
                logging.log(self._alert_level_to_log_level(alert_level), message)
            except queue.Full:
                logging.warning("Alert queue full - dropping alert")

    def _calculate_engine_health(self, datum, state):
        """Calculate overall engine health score (0-1)"""
        
        confidence = datum.get_derived_data(DerivedDataKey.RPM_CONFIDENCE) or 0.0
        harmonic_content = datum.get_derived_data(DerivedDataKey.HARMONIC_CONTENT) or 0.0
        rpm = datum.get_derived_data(DerivedDataKey.RPM) or 0.0
        
        health_factors = []
        
        # Confidence factor
        health_factors.append(confidence)
        
        # Harmonic content factor (healthy engines have good harmonics)
        harmonic_health = min(1.0, harmonic_content * 2.0)  # Scale 0-0.5 to 0-1
        health_factors.append(harmonic_health)
        
        # RPM stability factor
        if len(self.last_rpm_values) >= 10:
            rpm_std = np.std(self.last_rpm_values[-10:])
            rpm_mean = np.mean(self.last_rpm_values[-10:])
            if rpm_mean > 0:
                stability = 1.0 - min(1.0, rpm_std / (rpm_mean * 0.1))  # 10% variation = 0 health
                health_factors.append(stability)
        
        # State appropriateness factor
        if state in [AircraftEngineState.OVERSPEED, AircraftEngineState.ABNORMAL]:
            state_health = 0.0
        elif state == AircraftEngineState.SHUTDOWN:
            state_health = 1.0  # Shutdown is normal
        else:
            state_health = 0.8  # Normal operation
        health_factors.append(state_health)
        
        # Calculate overall health (geometric mean)
        if health_factors:
            health_score = np.power(np.prod(health_factors), 1.0/len(health_factors))
            return float(health_score)
        
        return 0.5

    def _alert_level_to_log_level(self, alert_level):
        """Convert alert level to logging level"""
        mapping = {
            AlertLevel.INFO: logging.INFO,
            AlertLevel.WARNING: logging.WARNING,
            AlertLevel.CAUTION: logging.WARNING,
            AlertLevel.CRITICAL: logging.CRITICAL
        }
        return mapping.get(alert_level, logging.INFO)

# Aircraft Feature Engineering Pipeline
class AircraftFeatureEngineeringPipeline:
    def __init__(self, engine_config):
        self.engine_config = engine_config
        self.__feature_engineering_blocks = []
        self.__output_blocks = []

    def add_block(self, block: FeatureExtraction) -> None:
        self.__feature_engineering_blocks.append(block)

    def add_output_block(self, block: FeatureExtraction) -> None:
        self.__output_blocks.append(block)

    def run(self, datum: Datum) -> Datum:
        start_time = time.time()
        
        try:
            # Run feature engineering blocks
            for block in self.__feature_engineering_blocks:
                datum = block.extract_features(datum)
            
            # Run output blocks
            for block in self.__output_blocks:
                datum = block.extract_features(datum)
            
            processing_time = time.time() - start_time
            logging.debug(f"Pipeline processing time: {processing_time:.3f}s")
            
        except Exception as e:
            logging.error(f"Aircraft pipeline error: {e}")
            # Set safe default values
            datum.set_derived_data(DerivedDataKey.ENGINE_STATE, AircraftEngineState.ABNORMAL.value)
            datum.set_derived_data(DerivedDataKey.RPM, 0.0)
            datum.set_derived_data(DerivedDataKey.RPM_CONFIDENCE, 0.0)
        
        return datum

# Utility Functions
def get_timestamp():
    """Return timestamp as milliseconds (uint64_t) for precision"""
    return int(time.time() * 1000)

def calculate_crc8(data):
    """Calculate CRC8 checksum"""
    crc = 0
    for byte in data:
        crc ^= byte
        for _ in range(8):
            if crc & 0x80:
                crc = (crc << 1) ^ 0x07
            else:
                crc = crc << 1
            crc &= 0xFF
    return crc

def create_aircraft_uart_packet(timestamp, datum: Datum, seq_num):
    """Create UART packet with aircraft-specific data"""
    try:
        rpm = datum.get_derived_data(DerivedDataKey.RPM) or 0.0
        rpm_percent = datum.get_derived_data(DerivedDataKey.RPM_PERCENT) or 0.0
        engine_state = datum.get_derived_data(DerivedDataKey.ENGINE_STATE) or 0
        rpm_confidence = datum.get_derived_data(DerivedDataKey.RPM_CONFIDENCE) or 0.0
        health_score = datum.get_derived_data(DerivedDataKey.ENGINE_HEALTH_SCORE) or 0.0
        peak_freq = datum.get_derived_data(DerivedDataKey.FREQUENCY_PEAK) or 0.0
        
        # Aircraft packet payload (25 bytes)
        payload = bytearray()
        payload.extend(struct.pack('<H', seq_num))           # 2-byte sequence number
        payload.extend(struct.pack('<Q', timestamp))         # 8-byte timestamp
        payload.extend(struct.pack('<f', float(rpm)))        # 4-byte RPM
        payload.extend(struct.pack('<f', float(rpm_percent))) # 4-byte RPM percent
        payload.extend(struct.pack('<B', int(engine_state))) # 1-byte engine state
        payload.extend(struct.pack('<f', float(rpm_confidence))) # 4-byte confidence
        payload.extend(struct.pack('<f', float(health_score))) # 4-byte health score
        payload.extend(struct.pack('<f', float(peak_freq)))  # 4-byte frequency
        
        length = len(payload)
        crc = calculate_crc8(payload)
        
        packet = bytearray([START_BYTE_DATA, length])
        packet.extend(payload)
        packet.append(crc)
        packet.append(END_BYTE)
        
        return packet
        
    except Exception as e:
        logging.error(f"Aircraft packet creation error: {e}")
        return None

def create_alert_packet(alert_level, message):
    """Create alert packet for aircraft warnings"""
    try:
        timestamp = get_timestamp()
        message_bytes = message.encode('utf-8')[:100]  # Limit message length
        
        payload = bytearray()
        payload.extend(struct.pack('<Q', timestamp))         # 8-byte timestamp
        payload.extend(struct.pack('<B', alert_level.value)) # 1-byte alert level
        payload.extend(struct.pack('<B', len(message_bytes))) # 1-byte message length
        payload.extend(message_bytes)                        # Variable message
        
        length = len(payload)
        crc = calculate_crc8(payload)
        
        packet = bytearray([START_BYTE_ALERT, length])
        packet.extend(payload)
        packet.append(crc)
        packet.append(END_BYTE)
        
        return packet
        
    except Exception as e:
        logging.error(f"Alert packet creation error: {e}")
        return None

# Enhanced UART Functions
def init_uart():
    """Initialize UART with aircraft-specific error handling"""
    global ser
    
    # Check if serial is available
    if not SERIAL_AVAILABLE:
        logging.warning("PySerial not available - running without UART")
        return False
    
    if not hasattr(serial, 'Serial'):
        logging.error("Serial.Serial class not found - check PySerial installation")
        return False
    
    max_attempts = 5
    for attempt in range(max_attempts):
        try:
            ser = serial.Serial(
                UART_PORT, 
                UART_BAUDRATE, 
                timeout=1, 
                rtscts=False, 
                xonxoff=False, 
                write_timeout=2.0
            )
            time.sleep(2)
            ser.reset_input_buffer()
            ser.reset_output_buffer()
            
            logging.info(f"✓ Aircraft UART connected on {UART_PORT} at {UART_BAUDRATE} baud")
            return True
            
        except serial.SerialException as e:
            logging.error(f"✗ UART connection failed (attempt {attempt + 1}/{max_attempts}): {e}")
            ser = None
            time.sleep(5)
        except Exception as e:
            logging.error(f"✗ Unexpected UART error: {e}")
            ser = None
            time.sleep(5)
    
    logging.critical("✗ Could not initialize UART after all attempts")
    return False

def send_aircraft_uart_packet(timestamp, datum: Datum, seq_num):
    """Send aircraft data packet via UART with enhanced reliability"""
    global packet_seq_num
    
    # Check UART connection
    if ser is None or not ser.is_open:
        logging.warning("UART disconnected. Attempting to reconnect...")
        init_uart()
        if ser is None:
            return False
    
    with uart_lock:
        for attempt in range(MAX_RETRIES):
            try:
                packet = create_aircraft_uart_packet(timestamp, datum, seq_num)
                if packet is None:
                    return False
                
                ser.reset_input_buffer()
                ser.reset_output_buffer()
                bytes_written = ser.write(packet)
                ser.flush()
                
                if bytes_written != len(packet):
                    logging.warning(f"✗ Incomplete write: {bytes_written}/{len(packet)} bytes")
                    continue
                
                # Wait for 3-byte ACK (0x06 + 2-byte seq_num)
                ser.timeout = ACK_TIMEOUT
                response = ser.read(3)
                
                if len(response) == 3 and response[0] == 0x06:
                    ack_seq = struct.unpack('<H', response[1:3])[0]
                    if ack_seq == seq_num:
                        rpm = datum.get_derived_data(DerivedDataKey.RPM) or 0.0
                        state_name = datum.get_derived_data(DerivedDataKey.ENGINE_STATE_NAME) or "UNKNOWN"
                        health = datum.get_derived_data(DerivedDataKey.ENGINE_HEALTH_SCORE) or 0.0
                        
                        logging.info(f"✓ Aircraft UART OK - Seq:{seq_num}, RPM:{rpm:.1f}, "
                                   f"State:{state_name}, Health:{health:.2f}")
                        return True
                    else:
                        logging.warning(f"✗ ACK sequence mismatch: expected {seq_num}, got {ack_seq}")
                else:
                    logging.warning(f"✗ No/invalid ACK (attempt {attempt + 1}/{MAX_RETRIES})")
                    
            except Exception as e:
                logging.error(f"✗ Aircraft UART send error (attempt {attempt + 1}): {e}")
            
            if attempt < MAX_RETRIES - 1:
                time.sleep(INTER_PACKET_DELAY * (attempt + 1))
        
        logging.error(f"✗ Failed to send aircraft packet after {MAX_RETRIES} attempts")
        return False

def export_fft_data(datum: Datum, filename_base):
    """Export FFT data and analysis results to files"""
    
    try:
        timestamp = datum.timestamp
        fft_data = datum.get_derived_data(DerivedDataKey.FFT_DATA)
        fft_freqs = datum.get_derived_data(DerivedDataKey.FFT_FREQUENCIES)
        fft_metadata = datum.get_derived_data(DerivedDataKey.FFT_METADATA)
        spectral_analysis = datum.get_derived_data(DerivedDataKey.SPECTRAL_ANALYSIS)
        
        if not fft_data or not fft_freqs:
            logging.warning("No FFT data available for export")
            return False
        
        # Create export directory
        export_dir = os.path.join(AUDIO_DIR, "fft_exports")
        os.makedirs(export_dir, exist_ok=True)
        
        # Generate filenames with timestamp
        timestamp_str = time.strftime("%Y%m%d_%H%M%S", time.localtime(timestamp))
        base_filename = f"{filename_base}_{timestamp_str}"
        
        # Export 1: CSV file with frequency and magnitude data
        csv_filename = os.path.join(export_dir, f"{base_filename}_fft.csv")
        with open(csv_filename, 'w', newline='') as csvfile:
            import csv
            writer = csv.writer(csvfile)
            
            # Header
            writer.writerow(['Frequency_Hz', 'Magnitude', 'Magnitude_dB'])
            
            # Data rows
            for freq, mag in zip(fft_freqs, fft_data):
                mag_db = 20 * np.log10(mag + 1e-10)  # Convert to dB
                writer.writerow([freq, mag, mag_db])
        
        # Export 2: JSON file with complete analysis
        json_filename = os.path.join(export_dir, f"{base_filename}_analysis.json")
        
        # Compile complete analysis data
        analysis_data = {
            'timestamp': timestamp,
            'timestamp_iso': time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(timestamp)),
            'aircraft_type': AIRCRAFT_TYPE,
            'engine_model': ENGINE_MODEL,
            
            # Engine analysis results
            'engine_analysis': {
                'rpm': datum.get_derived_data(DerivedDataKey.RPM),
                'rpm_percent': datum.get_derived_data(DerivedDataKey.RPM_PERCENT),
                'rpm_confidence': datum.get_derived_data(DerivedDataKey.RPM_CONFIDENCE),
                'rpm_trend': datum.get_derived_data(DerivedDataKey.RPM_TREND),
                'engine_state': datum.get_derived_data(DerivedDataKey.ENGINE_STATE),
                'engine_state_name': datum.get_derived_data(DerivedDataKey.ENGINE_STATE_NAME),
                'health_score': datum.get_derived_data(DerivedDataKey.ENGINE_HEALTH_SCORE)
            },
            
            # Frequency analysis results
            'frequency_analysis': {
                'peak_frequency': datum.get_derived_data(DerivedDataKey.FREQUENCY_PEAK),
                'frequency_confidence': datum.get_derived_data(DerivedDataKey.FREQUENCY_CONFIDENCE),
                'harmonic_content': datum.get_derived_data(DerivedDataKey.HARMONIC_CONTENT),
                'multiple_peaks': datum.get_derived_data(DerivedDataKey.FREQUENCY_PEAKS)
            },
            
            # FFT metadata
            'fft_metadata': fft_metadata,
            
            # Spectral analysis features
            'spectral_features': spectral_analysis,
            
            # Data arrays (truncated for size)
            'fft_summary': {
                'frequency_range': [min(fft_freqs), max(fft_freqs)],
                'magnitude_range': [min(fft_data), max(fft_data)],
                'peak_frequency_index': int(np.argmax(fft_data)),
                'total_data_points': len(fft_data)
            }
        }
        
        import json
        with open(json_filename, 'w') as jsonfile:
            json.dump(analysis_data, jsonfile, indent=2, default=str)
        
        # Export 3: NumPy binary file (fastest loading)
        npz_filename = os.path.join(export_dir, f"{base_filename}_fft.npz")
        np.savez_compressed(npz_filename,
                           frequencies=np.array(fft_freqs),
                           magnitudes=np.array(fft_data),
                           timestamp=timestamp,
                           sample_rate=fft_metadata.get('sample_rate', 0),
                           fft_size=fft_metadata.get('fft_size', 0))
        
        logging.info(f"FFT data exported: {csv_filename}, {json_filename}, {npz_filename}")
        return True
        
    except Exception as e:
        logging.error(f"FFT export error: {e}")
        return False

def create_fft_summary_string(datum: Datum):
    """Create a compact string summary of FFT analysis for logging"""
    
    try:
        fft_metadata = datum.get_derived_data(DerivedDataKey.FFT_METADATA) or {}
        spectral_analysis = datum.get_derived_data(DerivedDataKey.SPECTRAL_ANALYSIS) or {}
        frequency_peaks = datum.get_derived_data(DerivedDataKey.FREQUENCY_PEAKS) or []
        
        # Extract key metrics
        freq_resolution = fft_metadata.get('frequency_resolution', 0)
        fft_size = fft_metadata.get('fft_size', 0)
        snr_estimate = spectral_analysis.get('snr_estimate', 0)
        spectral_centroid = spectral_analysis.get('spectral_centroid', 0)
        peak_to_avg = spectral_analysis.get('peak_to_avg_ratio', 0)
        
        # Top 3 frequency peaks
        top_peaks = frequency_peaks[:3] if frequency_peaks else []
        peaks_str = ", ".join([f"{freq:.1f}Hz({mag:.0f})" for freq, mag in top_peaks])
        
        summary = (f"FFT[{fft_size}pts,{freq_resolution:.2f}Hz/bin] "
                  f"SNR:{snr_estimate:.1f}dB, Centroid:{spectral_centroid:.1f}Hz, "
                  f"P2A:{peak_to_avg:.1f}, Peaks:[{peaks_str}]")
        
        return summary
        
    except Exception as e:
        logging.error(f"FFT summary creation error: {e}")
        return "FFT[error]"
    """Send any queued alerts"""
    try:
        while not alert_queue.empty():
            alert_time, alert_level, message = alert_queue.get_nowait()
            packet = create_alert_packet(alert_level, message)
            
            if packet and ser and ser.is_open:
                with uart_lock:
                    try:
                        ser.write(packet)
                        ser.flush()
                        logging.info(f"Alert sent: {alert_level.name} - {message}")
                    except Exception as e:
                        logging.error(f"Failed to send alert: {e}")
    except queue.Empty:
        pass

def send_record_alert(timestamp):
    """Send recording start alert"""
    if ser is None:
        return
    with uart_lock:
        try:
            payload = struct.pack('<Q', timestamp)
            crc = calculate_crc8(payload)
            packet = bytearray([START_BYTE_RECORD, len(payload)])
            packet.extend(payload)
            packet.append(crc)
            packet.append(END_BYTE)
            ser.write(packet)
            ser.flush()
            logging.debug(f"Record alert sent: {timestamp}")
        except Exception as e:
            logging.error(f"✗ Record alert error: {e}")

def send_sync_packet():
    """Send time synchronization packet"""
    if ser is None:
        return
    with uart_lock:
        try:
            current_time = get_timestamp()
            payload = struct.pack('<Q', current_time)
            crc = calculate_crc8(payload)
            packet = bytearray([START_BYTE_SYNC, len(payload)])
            packet.extend(payload)
            packet.append(crc)
            packet.append(END_BYTE)
            ser.write(packet)
            ser.flush()
            logging.debug(f"Time sync sent: {current_time}")
        except Exception as e:
            logging.error(f"✗ Sync error: {e}")

# File management functions (unchanged but with aircraft logging)
def check_disk_space():
    """Check disk space for aircraft operations"""
    try:
        disk = psutil.disk_usage(AUDIO_DIR)
        free_space = disk.free / (1024 ** 3)
        free_percent = disk.free / disk.total
        
        is_low = free_space < MIN_FREE_SPACE_GB or free_percent < DISK_SPACE_THRESHOLD
        
        if is_low:
            logging.warning(f"Low disk space: {free_space:.1f}GB ({free_percent*100:.1f}%)")
        
        return is_low
    except Exception as e:
        logging.error(f"✗ Disk space check error: {e}")
        return False

def cleanup_old_files():
    """Clean old recordings with aircraft logging"""
    logging.info("🧹 Cleaning old aircraft recordings...")
    deleted_count = 0
    
    try:
        files = [(f, os.path.getmtime(os.path.join(AUDIO_DIR, f))) 
                for f in os.listdir(AUDIO_DIR) 
                if os.path.isfile(os.path.join(AUDIO_DIR, f)) and f.endswith('.wav')]
        
        files.sort(key=lambda x: x[1])
        
        for file, _ in files:
            if not check_disk_space():
                break
            
            path = os.path.join(AUDIO_DIR, file)
            try:
                os.remove(path)
                deleted_count += 1
                logging.info(f"Deleted aircraft recording: {file}")
            except Exception as e:
                logging.error(f"✗ Delete error: {e}")
                
    except Exception as e:
        logging.error(f"✗ Aircraft cleanup error: {e}")
    
    logging.info(f"✓ Aircraft cleanup complete. Deleted {deleted_count} files.")
    return deleted_count > 0

# Aircraft Recorder Thread
def aircraft_recorder():
    """Enhanced recorder for aircraft operations"""
    global file_counter, last_cleanup, packet_seq_num
    
    logging.info(f"🛩️  Starting aircraft audio recorder (Type: {AIRCRAFT_TYPE})")
    
    while True:
        try:
            # Check disk space
            if check_disk_space():
                cleanup_old_files()
            
            # Handle queue overflow
            if analyze_queue.full():
                logging.warning(f"Analyze queue full ({analyze_queue.qsize()}/{ANALYZE_QUEUE_SIZE}). "
                              "Dropping oldest item.")
                try:
                    analyze_queue.get_nowait()
                except queue.Empty:
                    pass
            
            # Record audio
            record_start_time = get_timestamp()
            send_record_alert(record_start_time)
            
            filename = f"Aircraft_Rec{file_counter:06d}.wav"
            filepath = os.path.join(AUDIO_DIR, filename)
            
            logging.info(f"🎙️  Recording: {filename}")
            
            audio = sd.rec(
                int(RECORD_SECONDS * SAMPLE_RATE),
                samplerate=SAMPLE_RATE,
                channels=1,
                dtype='int16',
                device=MIC_DEVICE
            )
            sd.wait()
            
            wav.write(filepath, SAMPLE_RATE, audio.flatten())
            logging.info(f"💾 Saved: {filename}")
            
            # Queue for analysis
            analyze_queue.put((filepath, record_start_time, packet_seq_num))
            file_counter += 1
            packet_seq_num = (packet_seq_num + 1) % 65536
            
            # Periodic cleanup
            if time.time() - last_cleanup > CLEAN_INTERVAL_DAYS * 86400:
                threading.Thread(target=cleanup_old_files, daemon=True).start()
                last_cleanup = time.time()
                
        except Exception as e:
            logging.error(f"✗ Aircraft recording error: {e}")
            time.sleep(1)

# Aircraft Analyzer Thread
def aircraft_analyzer():
    """Enhanced analyzer for aircraft engines"""
    
    # Create aircraft-specific engine configuration
    if AIRCRAFT_TYPE == "PISTON":
        engine_config = PistonEngineConfig(ENGINE_MODEL)
    elif AIRCRAFT_TYPE == "TURBOPROP":
        engine_config = TurbopropEngineConfig(ENGINE_MODEL)
    else:
        logging.error(f"Unsupported aircraft type: {AIRCRAFT_TYPE}")
        return
    
    # Create aircraft processing pipeline
    pipeline = AircraftFeatureEngineeringPipeline(engine_config)
    pipeline.add_block(AircraftDecimation(target_rate=DECIMATED_RATE))
    pipeline.add_block(AircraftFrequencyAnalyzer(engine_config))
    pipeline.add_block(AircraftRPMCalculator(engine_config))
    pipeline.add_output_block(AircraftEngineStateDetector(engine_config))
    
    logging.info(f"🔧 Aircraft analyzer started for {engine_config.model_name}")
    
    while True:
        try:
            filepath, record_timestamp, seq_num = analyze_queue.get()
            
            logging.info(f"🔍 Analyzing: {os.path.basename(filepath)} "
                        f"(Queue: {analyze_queue.qsize()})")
            
            # Load and prepare audio
            sample_rate, data = wav.read(filepath)
            if len(data.shape) > 1:
                data = data.flatten()
            
            # Process through aircraft pipeline
            datum = Datum(audio_array=data, sample_rate=sample_rate)
            datum = pipeline.run(datum)
            
            # Send results via UART
            success = send_aircraft_uart_packet(record_timestamp, datum, seq_num)
            
            # Send any queued alerts
            send_alert_if_queued()
            
            # Log results with FFT data
            if success:
                rpm = datum.get_derived_data(DerivedDataKey.RPM) or 0.0
                rpm_percent = datum.get_derived_data(DerivedDataKey.RPM_PERCENT) or 0.0
                state_name = datum.get_derived_data(DerivedDataKey.ENGINE_STATE_NAME) or "UNKNOWN"
                health = datum.get_derived_data(DerivedDataKey.ENGINE_HEALTH_SCORE) or 0.0
                confidence = datum.get_derived_data(DerivedDataKey.RPM_CONFIDENCE) or 0.0
                
                # Create FFT summary for logging
                fft_summary = create_fft_summary_string(datum)
                
                # Main engine status log
                logging.info(f"   ✈️  Aircraft Engine: RPM:{rpm:.1f} ({rpm_percent:.1f}%), "
                           f"State:{state_name}, Health:{health:.2f}, Conf:{confidence:.2f}")
                
                # Detailed FFT analysis log
                logging.info(f"   📊 {fft_summary}")
                
                # Export FFT data to files (every 10th analysis or on interesting conditions)
                should_export_fft = (
                    file_counter % 10 == 0 or  # Every 10th recording
                    health < 0.7 or  # Poor health score
                    state_name in ["OVERSPEED", "ABNORMAL"] or  # Abnormal states
                    confidence < 0.5  # Low confidence readings
                )
                
                if should_export_fft:
                    export_filename_base = f"Aircraft_{state_name}_seq{seq_num}"
                    export_success = export_fft_data(datum, export_filename_base)
                    if export_success:
                        logging.info(f"   💾 FFT data exported for analysis")
                
                # Additional spectral analysis logging for debugging
                spectral_analysis = datum.get_derived_data(DerivedDataKey.SPECTRAL_ANALYSIS) or {}
                if spectral_analysis:
                    fundamental_candidates = spectral_analysis.get('fundamental_candidates', [])
                    if len(fundamental_candidates) > 1:
                        # Log multiple frequency candidates
                        candidates_str = ", ".join([
                            f"{cand['frequency']:.1f}Hz({cand['magnitude']:.0f})" 
                            for cand in fundamental_candidates[:3]
                        ])
                        logging.debug(f"   🎵 Frequency candidates: {candidates_str}")
                
                # Log frequency analysis details
                peak_freq = datum.get_derived_data(DerivedDataKey.FREQUENCY_PEAK) or 0.0
                harmonic_content = datum.get_derived_data(DerivedDataKey.HARMONIC_CONTENT) or 0.0
                freq_confidence = datum.get_derived_data(DerivedDataKey.FREQUENCY_CONFIDENCE) or 0.0
                
                logging.debug(f"   🔊 Frequency Analysis: Peak:{peak_freq:.1f}Hz, "
                            f"Harmonics:{harmonic_content:.2f}, FreqConf:{freq_confidence:.2f}")
                
                
        except Exception as e:
            logging.error(f"Aircraft analysis error: {e}")

# Main function for aircraft system
def main():
    """Main function for aircraft engine monitoring system"""
    
    logging.info("🛩️  Aircraft Engine Monitoring System Starting...")
    logging.info(f"Aircraft Type: {AIRCRAFT_TYPE}")
    logging.info(f"Engine Model: {ENGINE_MODEL}")
    logging.info(f"Recordings directory: {AUDIO_DIR}")
    logging.info(f"Sample Rate: {SAMPLE_RATE}Hz, Decimated: {DECIMATED_RATE}Hz")
    
    # Initialize UART
    uart_success = init_uart()
    if not uart_success:
        logging.warning("System starting without UART connection")
    
    # Send initial sync
    send_sync_packet()
    
    # Start threads
    recorder_thread = threading.Thread(target=aircraft_recorder, daemon=True, name="Recorder")
    analyzer_thread = threading.Thread(target=aircraft_analyzer, daemon=True, name="Analyzer")
    
    recorder_thread.start()
    analyzer_thread.start()
    
    logging.info("✅ Aircraft system started!")
    logging.info("Press Ctrl+C to stop")
    
    try:
        while True:
            time.sleep(300)  # 5 minutes
            send_sync_packet()
            
            # Log system status
            queue_size = analyze_queue.qsize()
            if queue_size > ANALYZE_QUEUE_SIZE * 0.8:
                logging.warning(f"Analyze queue getting full: {queue_size}/{ANALYZE_QUEUE_SIZE}")
                
    except KeyboardInterrupt:
        logging.info("\n🛑 Stopping aircraft system...")
        if ser:
            ser.close()
        logging.info("✅ Aircraft system stopped")

if __name__ == "__main__":
    main()