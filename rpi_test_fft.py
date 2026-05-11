import serial
import psutil
import logging
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
import json
import csv

# Settings
RECORD_SECONDS = 5
SAMPLE_RATE = 44100
AUDIO_DIR = "./recordings"
FFT_EXPORT_DIR = "./fft_exports"  # New: FFT export directory
# UART_PORT = "/dev/serial1"  # Change to your ESP32 port (e.g., "/dev/ttyUSB0")
UART_BAUDRATE = 230400
CLEAN_INTERVAL_DAYS = 7
DECIMATED_RATE = 500
EVENTS_PER_CYCLE = 2
DISK_SPACE_THRESHOLD = 0.1
MIN_FREE_SPACE_GB = 1.0
ANALYZE_QUEUE_SIZE = 10
MIC_DEVICE = None
LOG_FILE = "audio_system.log"

# FFT Export Settings
EXPORT_FFT_EVERY_N = 5  # Export FFT data every 5th recording
EXPORT_ON_ANOMALY = True  # Export when unusual conditions detected
FFT_FREQUENCY_RANGE = (10, 200)  # Frequency range to analyze (Hz)

# UART Protocol Constants
START_BYTE_DATA = 0xAA
START_BYTE_RECORD = 0xAC
START_BYTE_SYNC = 0xAB
END_BYTE = 0x55
ACK_TIMEOUT = 1.0
MAX_RETRIES = 5
INTER_PACKET_DELAY = 0.05

# Setup logging
logging.basicConfig(filename=LOG_FILE, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Create directories
os.makedirs(AUDIO_DIR, exist_ok=True)
os.makedirs(FFT_EXPORT_DIR, exist_ok=True)

# Queues
analyze_queue = queue.Queue(maxsize=ANALYZE_QUEUE_SIZE)

# Serial UART init with better error handling
ser = None
uart_lock = threading.Lock()

file_counter = 1
last_cleanup = time.time()
packet_seq_num = 0

# Enhanced Datum classes with FFT export support
class RawDatumKey:
    AUDIO_ARRAY = "audio_array"
    SAMPLE_RATE = "sample_rate"

class DerivedDataKey:
    DECIMATED_AUDIO = "decimated_audio"
    DECIMATED_AUDIO_RATE = "decimated_audio_rate"
    FFT_DATA = "fft_data"
    FFT_FREQUENCIES = "fft_frequencies"  # New: Frequency axis
    FFT_METADATA = "fft_metadata"  # New: FFT analysis metadata
    SPECTRAL_FEATURES = "spectral_features"  # New: Advanced spectral analysis
    FREQUENCY_PEAK = "frequency_peak"
    DECIMATED_FREQUENCY_PEAK = "decimated_frequency_peak"
    FREQUENCY_PEAKS_LIST = "frequency_peaks_list"  # New: Multiple peaks
    RPM = "rpm"
    ENGINE_STATUS = "engine_status"

class Datum:
    def __init__(self, audio_array, sample_rate):
        self._raw_data = {RawDatumKey.AUDIO_ARRAY: audio_array, RawDatumKey.SAMPLE_RATE: sample_rate}
        self._derived_data = {}
        self.timestamp = time.time()  # Add timestamp for export
        self.recording_id = None  # Will be set during analysis

    def get_raw_datum(self, key):
        return self._raw_data.get(key)

    def set_raw_datum(self, key, value):
        self._raw_data[key] = value

    def get_derived_data(self, key):
        return self._derived_data.get(key)

    def set_derived_data(self, key, value):
        self._derived_data[key] = value

# FeatureExtraction interface
class FeatureExtraction:
    def extract_features(self, datum: Datum) -> Datum:
        raise NotImplementedError

# Enhanced Decimation class
class Decimation(FeatureExtraction):
    def __init__(self, target_rate=500):
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
                decimated_signal = decimate(signal, decimation_factor, axis=0, ftype='iir')
                decimated_rate = original_rate // decimation_factor
            datum.set_derived_data(DerivedDataKey.DECIMATED_AUDIO, decimated_signal)
            datum.set_derived_data(DerivedDataKey.DECIMATED_AUDIO_RATE, decimated_rate)
        except Exception as e:
            print(f"✗ Decimation error: {e}")
            logging.error(f"Decimation error: {e}")
            datum.set_derived_data(DerivedDataKey.DECIMATED_AUDIO, signal)
            datum.set_derived_data(DerivedDataKey.DECIMATED_AUDIO_RATE, original_rate)
        return datum

# Enhanced FrequencyPeakFinder with comprehensive FFT export
class FrequencyPeakFinder(FeatureExtraction):
    def __init__(self, buffer_seconds=5):
        self.buffer_seconds = buffer_seconds

    def extract_features(self, datum: Datum):
        try:
            # Process original signal
            self._analyze_signal(datum, use_decimated=False)
            
            # Process decimated signal with full FFT export
            self._analyze_signal(datum, use_decimated=True)
            
        except Exception as e:
            print(f"✗ Peak finding error: {e}")
            logging.error(f"Peak finding error: {e}")
            self._set_default_values(datum)
        return datum

    def _analyze_signal(self, datum: Datum, use_decimated=True):
        """Analyze signal with comprehensive FFT data export"""
        
        if use_decimated:
            signal = datum.get_derived_data(DerivedDataKey.DECIMATED_AUDIO)
            sample_rate = datum.get_derived_data(DerivedDataKey.DECIMATED_AUDIO_RATE)
        else:
            signal = datum.get_raw_datum(RawDatumKey.AUDIO_ARRAY)
            sample_rate = datum.get_raw_datum(RawDatumKey.SAMPLE_RATE)

        if signal is None or len(signal) == 0:
            return

        # Apply windowing
        windowed_signal = signal * np.hanning(len(signal))
        
        # Zero-padding for better frequency resolution
        N_original = len(windowed_signal)
        N_padded = 2 ** int(np.ceil(np.log2(N_original)))
        if N_padded > N_original:
            padded_signal = np.pad(windowed_signal, (0, N_padded - N_original), mode='constant')
        else:
            padded_signal = windowed_signal
        
        # Compute FFT
        N = len(padded_signal)
        fft_complex = fftpack.fft(padded_signal)
        fft_data = np.abs(fft_complex[:N//2])
        freqs = fftpack.fftfreq(N, 1/sample_rate)[:N//2]

        # Store complete FFT data (for decimated signal only to save space)
        if use_decimated:
            # Create comprehensive FFT metadata
            fft_metadata = {
                'sample_rate': sample_rate,
                'original_length': N_original,
                'padded_length': N,
                'window_type': 'hanning',
                'frequency_resolution': sample_rate / N,
                'nyquist_frequency': sample_rate / 2,
                'zero_padded': N_padded > N_original,
                'analysis_timestamp': time.time(),
                'decimation_factor': SAMPLE_RATE // sample_rate if sample_rate != SAMPLE_RATE else 1
            }
            
            # Store full FFT data and frequencies
            datum.set_derived_data(DerivedDataKey.FFT_DATA, fft_data.tolist())
            datum.set_derived_data(DerivedDataKey.FFT_FREQUENCIES, freqs.tolist())
            datum.set_derived_data(DerivedDataKey.FFT_METADATA, fft_metadata)
            
            # Calculate advanced spectral features
            spectral_features = self._calculate_spectral_features(fft_data, freqs, sample_rate)
            datum.set_derived_data(DerivedDataKey.SPECTRAL_FEATURES, spectral_features)

        # Analyze frequency range of interest
        min_freq, max_freq = FFT_FREQUENCY_RANGE
        valid_range = (freqs >= min_freq) & (freqs <= max_freq)
        
        if np.any(valid_range):
            valid_fft = fft_data[valid_range]
            valid_freqs = freqs[valid_range]
            
            # Find primary peak
            peak_idx = np.argmax(valid_fft)
            peak_freq = valid_freqs[peak_idx]
            
            # Find multiple peaks for comprehensive analysis
            multiple_peaks = self._find_multiple_peaks(valid_fft, valid_freqs)
            
            if use_decimated:
                datum.set_derived_data(DerivedDataKey.DECIMATED_FREQUENCY_PEAK, peak_freq)
                datum.set_derived_data(DerivedDataKey.FREQUENCY_PEAKS_LIST, multiple_peaks)
            else:
                datum.set_derived_data(DerivedDataKey.FREQUENCY_PEAK, peak_freq)
        else:
            if use_decimated:
                datum.set_derived_data(DerivedDataKey.DECIMATED_FREQUENCY_PEAK, 0.0)
                datum.set_derived_data(DerivedDataKey.FREQUENCY_PEAKS_LIST, [])
            else:
                datum.set_derived_data(DerivedDataKey.FREQUENCY_PEAK, 0.0)

    def _find_multiple_peaks(self, fft_data, freqs):
        """Find multiple frequency peaks for detailed analysis"""
        try:
            # Use scipy's find_peaks to identify multiple peaks
            prominence_threshold = np.max(fft_data) * 0.1  # 10% of max
            peaks, properties = find_peaks(fft_data, 
                                         prominence=prominence_threshold,
                                         distance=5)  # Minimum 5 bins apart
            
            # Extract peak information
            peak_list = []
            for i, peak_idx in enumerate(peaks):
                peak_info = {
                    'frequency': float(freqs[peak_idx]),
                    'magnitude': float(fft_data[peak_idx]),
                    'magnitude_db': float(20 * np.log10(fft_data[peak_idx] + 1e-10)),
                    'prominence': float(properties['prominences'][i]) if 'prominences' in properties else 0.0
                }
                peak_list.append(peak_info)
            
            # Sort by magnitude (highest first) and return top 10
            peak_list.sort(key=lambda x: x['magnitude'], reverse=True)
            return peak_list[:10]
            
        except Exception as e:
            logging.error(f"Multiple peaks detection error: {e}")
            return []

    def _calculate_spectral_features(self, fft_data, freqs, sample_rate):
        """Calculate advanced spectral analysis features"""
        try:
            features = {}
            
            # Spectral centroid (brightness)
            if np.sum(fft_data) > 0:
                features['spectral_centroid'] = float(np.sum(freqs * fft_data) / np.sum(fft_data))
            else:
                features['spectral_centroid'] = 0.0
            
            # Spectral spread (bandwidth)
            centroid = features['spectral_centroid']
            if np.sum(fft_data) > 0:
                features['spectral_spread'] = float(np.sqrt(
                    np.sum(((freqs - centroid) ** 2) * fft_data) / np.sum(fft_data)
                ))
            else:
                features['spectral_spread'] = 0.0
            
            # Spectral rolloff (95% energy point)
            cumulative_energy = np.cumsum(fft_data ** 2)
            total_energy = cumulative_energy[-1]
            if total_energy > 0:
                rolloff_idx = np.where(cumulative_energy >= 0.95 * total_energy)[0]
                features['spectral_rolloff'] = float(freqs[rolloff_idx[0]]) if len(rolloff_idx) > 0 else float(freqs[-1])
            else:
                features['spectral_rolloff'] = 0.0
            
            # Focus on analysis band
            min_freq, max_freq = FFT_FREQUENCY_RANGE
            analysis_range = (freqs >= min_freq) & (freqs <= max_freq)
            
            if np.any(analysis_range):
                analysis_fft = fft_data[analysis_range]
                
                # Signal-to-noise ratio in analysis band
                noise_floor = np.percentile(analysis_fft, 10)  # 10th percentile as noise
                signal_peak = np.max(analysis_fft)
                features['snr_db'] = float(20 * np.log10((signal_peak + 1e-10) / (noise_floor + 1e-10)))
                
                # Peak-to-average ratio
                avg_magnitude = np.mean(analysis_fft)
                features['peak_to_avg_ratio'] = float(signal_peak / (avg_magnitude + 1e-10))
                
                # Total energy in analysis band
                features['analysis_band_energy'] = float(np.sum(analysis_fft ** 2))
                
                # Spectral flatness (measure of noisiness)
                geometric_mean = np.exp(np.mean(np.log(analysis_fft + 1e-10)))
                arithmetic_mean = np.mean(analysis_fft)
                features['spectral_flatness'] = float(geometric_mean / (arithmetic_mean + 1e-10))
            else:
                features.update({
                    'snr_db': 0.0,
                    'peak_to_avg_ratio': 0.0,
                    'analysis_band_energy': 0.0,
                    'spectral_flatness': 0.0
                })
            
            # Total signal energy
            features['total_energy'] = float(np.sum(fft_data ** 2))
            
            # Dynamic range
            features['dynamic_range_db'] = float(20 * np.log10(
                (np.max(fft_data) + 1e-10) / (np.min(fft_data[fft_data > 0]) + 1e-10)
            )) if np.any(fft_data > 0) else 0.0
            
            return features
            
        except Exception as e:
            logging.error(f"Spectral features calculation error: {e}")
            return {
                'spectral_centroid': 0.0,
                'spectral_spread': 0.0,
                'spectral_rolloff': 0.0,
                'snr_db': 0.0,
                'peak_to_avg_ratio': 0.0,
                'analysis_band_energy': 0.0,
                'spectral_flatness': 0.0,
                'total_energy': 0.0,
                'dynamic_range_db': 0.0
            }

    def _set_default_values(self, datum):
        """Set default values on error"""
        datum.set_derived_data(DerivedDataKey.FREQUENCY_PEAK, 0.0)
        datum.set_derived_data(DerivedDataKey.DECIMATED_FREQUENCY_PEAK, 0.0)
        datum.set_derived_data(DerivedDataKey.FFT_DATA, [])
        datum.set_derived_data(DerivedDataKey.FFT_FREQUENCIES, [])
        datum.set_derived_data(DerivedDataKey.FFT_METADATA, {})
        datum.set_derived_data(DerivedDataKey.SPECTRAL_FEATURES, {})
        datum.set_derived_data(DerivedDataKey.FREQUENCY_PEAKS_LIST, [])

# Enhanced RPM class
class RPM(FeatureExtraction):
    def __init__(self, events_per_crankshaft_cycle=2):
        self.events_per_cycle = events_per_crankshaft_cycle

    def extract_features(self, datum: Datum):
        try:
            peak_freq = datum.get_derived_data(DerivedDataKey.DECIMATED_FREQUENCY_PEAK)
            if peak_freq is None:
                peak_freq = 0.0
            rpm = (peak_freq * 60) / self.events_per_cycle
            if rpm < 0 or rpm > 10000:
                rpm = 0.0
            datum.set_derived_data(DerivedDataKey.RPM, rpm)
            engine_status = 1 if rpm > 100 else 0
            datum.set_derived_data(DerivedDataKey.ENGINE_STATUS, engine_status)
        except Exception as e:
            print(f"✗ RPM calculation error: {e}")
            logging.error(f"RPM calculation error: {e}")
            datum.set_derived_data(DerivedDataKey.RPM, 0.0)
            datum.set_derived_data(DerivedDataKey.ENGINE_STATUS, 0)
        return datum

# FFT Export Functions
def export_fft_data(datum: Datum, recording_filename):
    """Export comprehensive FFT data and analysis results"""
    try:
        # Get FFT data
        fft_data = datum.get_derived_data(DerivedDataKey.FFT_DATA)
        fft_frequencies = datum.get_derived_data(DerivedDataKey.FFT_FREQUENCIES)
        fft_metadata = datum.get_derived_data(DerivedDataKey.FFT_METADATA)
        spectral_features = datum.get_derived_data(DerivedDataKey.SPECTRAL_FEATURES)
        
        if not fft_data or not fft_frequencies:
            logging.warning(f"No FFT data available for export: {recording_filename}")
            return False
        
        # Create timestamp-based filename
        timestamp_str = time.strftime("%Y%m%d_%H%M%S", time.localtime(datum.timestamp))
        base_filename = f"{os.path.splitext(recording_filename)[0]}_{timestamp_str}"
        
        # Export 1: CSV file with frequency and magnitude data
        csv_filename = os.path.join(FFT_EXPORT_DIR, f"{base_filename}_fft.csv")
        with open(csv_filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Frequency_Hz', 'Magnitude', 'Magnitude_dB', 'Phase_Radians'])
            
            for i, (freq, mag) in enumerate(zip(fft_frequencies, fft_data)):
                mag_db = 20 * np.log10(mag + 1e-10)
                # Phase would require storing complex FFT data, set to 0 for now
                phase = 0.0
                writer.writerow([freq, mag, mag_db, phase])
        
        # Export 2: Comprehensive JSON analysis file
        json_filename = os.path.join(FFT_EXPORT_DIR, f"{base_filename}_analysis.json")
        
        analysis_data = {
            'recording_info': {
                'filename': recording_filename,
                'timestamp': datum.timestamp,
                'timestamp_iso': time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(datum.timestamp)),
                'recording_id': datum.recording_id
            },
            
            'audio_parameters': {
                'sample_rate': datum.get_raw_datum(RawDatumKey.SAMPLE_RATE),
                'recording_seconds': RECORD_SECONDS,
                'decimated_rate': datum.get_derived_data(DerivedDataKey.DECIMATED_AUDIO_RATE)
            },
            
            'engine_analysis': {
                'rpm': datum.get_derived_data(DerivedDataKey.RPM),
                'engine_status': datum.get_derived_data(DerivedDataKey.ENGINE_STATUS),
                'peak_frequency': datum.get_derived_data(DerivedDataKey.FREQUENCY_PEAK),
                'decimated_peak_frequency': datum.get_derived_data(DerivedDataKey.DECIMATED_FREQUENCY_PEAK),
                'events_per_cycle': EVENTS_PER_CYCLE
            },
            
            'frequency_analysis': {
                'multiple_peaks': datum.get_derived_data(DerivedDataKey.FREQUENCY_PEAKS_LIST),
                'analysis_range_hz': FFT_FREQUENCY_RANGE
            },
            
            'fft_metadata': fft_metadata,
            'spectral_features': spectral_features,
            
            'fft_summary': {
                'total_bins': len(fft_data),
                'frequency_range_hz': [min(fft_frequencies), max(fft_frequencies)],
                'magnitude_range': [min(fft_data), max(fft_data)],
                'max_magnitude_frequency': fft_frequencies[np.argmax(fft_data)],
                'mean_magnitude': np.mean(fft_data),
                'std_magnitude': np.std(fft_data)
            }
        }
        
        with open(json_filename, 'w') as jsonfile:
            json.dump(analysis_data, jsonfile, indent=2, default=str)
        
        # Export 3: NumPy compressed file for fast loading
        npz_filename = os.path.join(FFT_EXPORT_DIR, f"{base_filename}_fft.npz")
        np.savez_compressed(npz_filename,
                           frequencies=np.array(fft_frequencies),
                           magnitudes=np.array(fft_data),
                           timestamp=datum.timestamp,
                           sample_rate=fft_metadata.get('sample_rate', 0),
                           recording_filename=recording_filename)
        
        # Export 4: Analysis band only (focused data)
        min_freq, max_freq = FFT_FREQUENCY_RANGE
        freq_array = np.array(fft_frequencies)
        mag_array = np.array(fft_data)
        analysis_mask = (freq_array >= min_freq) & (freq_array <= max_freq)
        
        if np.any(analysis_mask):
            analysis_csv = os.path.join(FFT_EXPORT_DIR, f"{base_filename}_analysis_band.csv")
            with open(analysis_csv, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['Frequency_Hz', 'Magnitude', 'Magnitude_dB'])
                
                analysis_freqs = freq_array[analysis_mask]
                analysis_mags = mag_array[analysis_mask]
                
                for freq, mag in zip(analysis_freqs, analysis_mags):
                    mag_db = 20 * np.log10(mag + 1e-10)
                    writer.writerow([freq, mag, mag_db])
        
        logging.info(f"FFT data exported: {base_filename} (CSV, JSON, NPZ, Analysis Band)")
        print(f"📊 FFT exported: {base_filename}")
        return True
        
    except Exception as e:
        logging.error(f"FFT export error for {recording_filename}: {e}")
        print(f"✗ FFT export failed: {e}")
        return False

def should_export_fft(datum: Datum, recording_count):
    """Determine if FFT data should be exported for this recording"""
    
    # Always export every Nth recording
    if recording_count % EXPORT_FFT_EVERY_N == 0:
        return True, "Regular interval export"
    
    if not EXPORT_ON_ANOMALY:
        return False, "Anomaly detection disabled"
    
    # Export on anomalous conditions
    rpm = datum.get_derived_data(DerivedDataKey.RPM) or 0.0
    spectral_features = datum.get_derived_data(DerivedDataKey.SPECTRAL_FEATURES) or {}
    
    # High RPM
    if rpm > 8000:
        return True, f"High RPM: {rpm:.1f}"
    
    # Very low SNR
    snr = spectral_features.get('snr_db', 0)
    if snr < 3:
        return True, f"Low SNR: {snr:.1f}dB"
    
    # Poor peak definition
    p2a = spectral_features.get('peak_to_avg_ratio', 0)
    if p2a < 1.5:
        return True, f"Poor peak definition: {p2a:.1f}"
    
    # Very low or high spectral centroid (unusual frequency distribution)
    centroid = spectral_features.get('spectral_centroid', 0)
    if centroid < 20 or centroid > 150:
        return True, f"Unusual spectral centroid: {centroid:.1f}Hz"
    
    return False, "Normal conditions"

def create_fft_summary_log(datum: Datum):
    """Create a concise FFT summary for logging"""
    try:
        fft_metadata = datum.get_derived_data(DerivedDataKey.FFT_METADATA) or {}
        spectral_features = datum.get_derived_data(DerivedDataKey.SPECTRAL_FEATURES) or {}
        peaks_list = datum.get_derived_data(DerivedDataKey.FREQUENCY_PEAKS_LIST) or []
        
        # Key metrics
        freq_res = fft_metadata.get('frequency_resolution', 0)
        snr = spectral_features.get('snr_db', 0)
        centroid = spectral_features.get('spectral_centroid', 0)
        p2a = spectral_features.get('peak_to_avg_ratio', 0)
        
        # Top 3 peaks
        top_peaks = peaks_list[:3]
        peaks_str = ", ".join([f"{p['frequency']:.1f}Hz({p['magnitude']:.0f})" for p in top_peaks])
        
        summary = (f"FFT[{freq_res:.2f}Hz/bin] SNR:{snr:.1f}dB, "
                  f"Centroid:{centroid:.1f}Hz, P2A:{p2a:.1f}, "
                  f"Peaks:[{peaks_str}]")
        
        return summary
    except Exception as e:
        logging.error(f"FFT summary creation error: {e}")
        return "FFT[error]"

# Enhanced FeatureEngineeringPipeline
class FeatureEngineeringPipeline:
    def __init__(self):
        self.__feature_engineering_blocks = []
        self.__output_blocks = []

    def add_block(self, block: FeatureExtraction) -> None:
        self.__feature_engineering_blocks.append(block)

    def add_output_block(self, block: FeatureExtraction) -> None:
        self.__output_blocks.append(block)

    def run(self, datum: Datum) -> Datum:
        for block in self.__feature_engineering_blocks:
            datum = block.extract_features(datum)
        for block in self.__output_blocks:
            datum = block.extract_features(datum)
        return datum

# Utility functions (unchanged)
def get_timestamp():
    return int(time.time() * 1000)

def calculate_crc8(data):
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

def create_uart_packet(timestamp, datum: Datum, seq_num):
    try:
        rpm = datum.get_derived_data(DerivedDataKey.RPM) or 0.0
        engine_status = datum.get_derived_data(DerivedDataKey.ENGINE_STATUS) or 0
        peak_freq = datum.get_derived_data(DerivedDataKey.FREQUENCY_PEAK) or 0.0
        payload = bytearray()
        payload.extend(struct.pack('<H', seq_num))
        payload.extend(struct.pack('<Q', timestamp))
        payload.extend(struct.pack('<f', float(rpm)))
        payload.extend(struct.pack('<B', int(engine_status)))
        payload.extend(struct.pack('<f', float(peak_freq)))
        length = len(payload)
        crc = calculate_crc8(payload)
        packet = bytearray([START_BYTE_DATA, length])
        packet.extend(payload)
        packet.append(crc)
        packet.append(END_BYTE)
        return packet
    except Exception as e:
        print(f"Packet creation error: {e}")
        logging.error(f"Packet creation error: {e}")
        return None

def send_uart_packet(timestamp, datum: Datum, seq_num):
    # UART functionality disabled for this example
    # In real implementation, this would send the packet
    return True  # Simulate successful transmission

def send_record_alert(timestamp):
    # UART functionality disabled for this example
    pass

def send_sync_packet():
    # UART functionality disabled for this example
    pass

def check_disk_space():
    try:
        disk = psutil.disk_usage(AUDIO_DIR)
        free_space = disk.free / (1024 ** 3)
        free_percent = disk.free / disk.total
        return free_space < MIN_FREE_SPACE_GB or free_percent < DISK_SPACE_THRESHOLD
    except Exception as e:
        print(f"✗ Disk space check error: {e}")
        return False

def cleanup_old_files():
    print("🧹 Cleaning old recordings...")
    logging.info("Cleaning old recordings")
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
                print(f"Deleted: {file}")
                logging.info(f"Deleted: {file}")
            except Exception as e:
                print(f"✗ Delete error: {e}")
                logging.error(f"Delete error: {e}")
    except Exception as e:
        print(f"✗ Cleanup error: {e}")
        logging.error(f"Cleanup error: {e}")
    print(f"✓ Cleanup complete. Deleted {deleted_count} files.")
    logging.info(f"Cleanup complete. Deleted {deleted_count} files.")
    return deleted_count > 0

def recorder():
    global file_counter, last_cleanup, packet_seq_num
    print(f"Recording from default microphone")
    logging.info("Recording from default microphone")
    while True:
        try:
            if check_disk_space():
                cleanup_old_files()
            
            # Handle queue overflow
            if analyze_queue.full():
                print(f"Analyze queue full ({analyze_queue.qsize()}/{ANALYZE_QUEUE_SIZE}). Dropping oldest item.")
                logging.warning(f"Analyze queue full. Dropping oldest item.")
                try:
                    analyze_queue.get_nowait()
                except queue.Empty:
                    pass
            
            record_start_time = get_timestamp()
            send_record_alert(record_start_time)
            filename = f"Rec{file_counter:06d}.wav"
            filepath = os.path.join(AUDIO_DIR, filename)
            print(f"\n📹 Recording: {filename}")
            logging.info(f"Recording: {filename}")
            
            audio = sd.rec(
                int(RECORD_SECONDS * SAMPLE_RATE),
                samplerate=SAMPLE_RATE,
                channels=1,
                dtype='int16',
                device=MIC_DEVICE
            )
            sd.wait()
            wav.write(filepath, SAMPLE_RATE, audio.flatten())
            print(f"💾 Saved: {filename}")
            logging.info(f"Saved: {filename}")
            
            analyze_queue.put((filepath, record_start_time, packet_seq_num, filename))
            file_counter += 1
            packet_seq_num = (packet_seq_num + 1) % 65536
            
            if time.time() - last_cleanup > CLEAN_INTERVAL_DAYS * 86400:
                threading.Thread(target=cleanup_old_files, daemon=True).start()
                last_cleanup = time.time()
                
        except Exception as e:
            print(f"✗ Recording error: {e}")
            logging.error(f"Recording error: {e}")
            time.sleep(1)

def analyzer():
    pipeline = FeatureEngineeringPipeline()
    pipeline.add_block(Decimation(target_rate=DECIMATED_RATE))
    pipeline.add_block(FrequencyPeakFinder(buffer_seconds=RECORD_SECONDS))
    pipeline.add_output_block(RPM(events_per_crankshaft_cycle=EVENTS_PER_CYCLE))
    
    recording_count = 0
    
    while True:
        try:
            filepath, record_timestamp, seq_num, filename = analyze_queue.get()
            recording_count += 1
            
            print(f"🔍 Analyzing: {os.path.basename(filepath)} (Queue: {analyze_queue.qsize()})")
            logging.info(f"Analyzing: {os.path.basename(filepath)} (Queue: {analyze_queue.qsize()})")
            
            # Load audio file
            sample_rate, data = wav.read(filepath)
            if len(data.shape) > 1:
                data = data.flatten()
            
            # Create datum and set recording ID
            datum = Datum(audio_array=data, sample_rate=sample_rate)
            datum.recording_id = f"REC_{recording_count:06d}"
            
            # Process through pipeline
            datum = pipeline.run(datum)
            
            # Send UART packet
            success = send_uart_packet(record_timestamp, datum, seq_num)
            
            if success:
                # Get analysis results
                rpm = datum.get_derived_data(DerivedDataKey.RPM) or 0.0
                status = datum.get_derived_data(DerivedDataKey.ENGINE_STATUS) or 0
                freq = datum.get_derived_data(DerivedDataKey.FREQUENCY_PEAK) or 0.0
                decimated_freq = datum.get_derived_data(DerivedDataKey.DECIMATED_FREQUENCY_PEAK) or 0.0
                
                # Create FFT summary for logging
                fft_summary = create_fft_summary_log(datum)
                
                # Main results log
                print(f"   ⚡ RPM:{rpm:.1f}, Status:{status}, Freq:{freq:.1f}Hz, DecFreq:{decimated_freq:.1f}Hz")
                print(f"   📊 {fft_summary}")
                
                # Detailed logging
                logging.info(f"Analysis results - RPM:{rpm:.1f}, Status:{status}, Freq:{freq:.1f}Hz")
                logging.info(f"FFT Summary - {fft_summary}")
                
                # Check if we should export FFT data
                should_export, export_reason = should_export_fft(datum, recording_count)
                
                if should_export:
                    export_success = export_fft_data(datum, filename)
                    if export_success:
                        print(f"   💾 FFT exported: {export_reason}")
                        logging.info(f"FFT data exported for {filename}: {export_reason}")
                    else:
                        print(f"   ❌ FFT export failed: {export_reason}")
                        logging.error(f"FFT export failed for {filename}: {export_reason}")
                
                # Log spectral features for debugging
                spectral_features = datum.get_derived_data(DerivedDataKey.SPECTRAL_FEATURES) or {}
                if spectral_features:
                    snr = spectral_features.get('snr_db', 0)
                    centroid = spectral_features.get('spectral_centroid', 0)
                    energy = spectral_features.get('analysis_band_energy', 0)
                    flatness = spectral_features.get('spectral_flatness', 0)
                    
                    logging.debug(f"Spectral features - SNR:{snr:.1f}dB, Centroid:{centroid:.1f}Hz, "
                                f"Energy:{energy:.0f}, Flatness:{flatness:.3f}")
                
                # Log multiple peaks if available
                peaks_list = datum.get_derived_data(DerivedDataKey.FREQUENCY_PEAKS_LIST) or []
                if len(peaks_list) > 1:
                    top_3_peaks = peaks_list[:3]
                    peaks_str = ", ".join([f"{p['frequency']:.1f}Hz({p['magnitude']:.0f})" 
                                         for p in top_3_peaks])
                    logging.debug(f"Top frequency peaks: {peaks_str}")
                
        except Exception as e:
            print(f"❌ Analysis error: {e}")
            logging.error(f"Analysis error: {e}")

def print_fft_export_summary():
    """Print summary of FFT export configuration"""
    print("\n📊 FFT Export Configuration:")
    print(f"   Export Directory: {FFT_EXPORT_DIR}")
    print(f"   Export Every: {EXPORT_FFT_EVERY_N} recordings")
    print(f"   Export on Anomaly: {EXPORT_ON_ANOMALY}")
    print(f"   Frequency Range: {FFT_FREQUENCY_RANGE[0]}-{FFT_FREQUENCY_RANGE[1]} Hz")
    print(f"   Decimated Rate: {DECIMATED_RATE} Hz")
    print("   Export Formats: CSV, JSON, NPZ, Analysis Band CSV")
    print("   Anomaly Triggers: High RPM, Low SNR, Poor Peaks, Unusual Spectral Centroid")
    print()

def analyze_existing_fft_exports():
    """Analyze existing FFT exports and print summary"""
    try:
        export_files = [f for f in os.listdir(FFT_EXPORT_DIR) if f.endswith('_analysis.json')]
        
        if not export_files:
            print("📊 No existing FFT exports found.")
            return
        
        print(f"📊 Found {len(export_files)} existing FFT exports:")
        
        rpms = []
        snrs = []
        centroids = []
        
        for json_file in export_files[:5]:  # Show details for first 5
            try:
                with open(os.path.join(FFT_EXPORT_DIR, json_file), 'r') as f:
                    data = json.load(f)
                
                rpm = data.get('engine_analysis', {}).get('rpm', 0)
                snr = data.get('spectral_features', {}).get('snr_db', 0)
                centroid = data.get('spectral_features', {}).get('spectral_centroid', 0)
                timestamp = data.get('recording_info', {}).get('timestamp_iso', 'Unknown')
                
                rpms.append(rpm)
                snrs.append(snr)
                centroids.append(centroid)
                
                print(f"   {json_file[:20]}... RPM:{rpm:.1f}, SNR:{snr:.1f}dB, "
                      f"Centroid:{centroid:.1f}Hz, Time:{timestamp}")
                
            except Exception as e:
                print(f"   Error reading {json_file}: {e}")
        
        if rpms:
            print(f"\n📈 Summary Statistics:")
            print(f"   RPM Range: {min(rpms):.1f} - {max(rpms):.1f} (avg: {np.mean(rpms):.1f})")
            print(f"   SNR Range: {min(snrs):.1f} - {max(snrs):.1f}dB (avg: {np.mean(snrs):.1f}dB)")
            print(f"   Centroid Range: {min(centroids):.1f} - {max(centroids):.1f}Hz (avg: {np.mean(centroids):.1f}Hz)")
        
    except Exception as e:
        print(f"❌ Error analyzing FFT exports: {e}")

def main():
    print("🎵 Enhanced Audio Processing System with FFT Export Starting...")
    logging.info("Enhanced Audio Processing System with FFT Export Starting")
    
    print(f"📁 Recordings directory: {AUDIO_DIR}")
    print(f"📊 FFT exports directory: {FFT_EXPORT_DIR}")
    
    # Print FFT export configuration
    print_fft_export_summary()
    
    # Analyze existing exports
    analyze_existing_fft_exports()
    
    # Start sync (disabled UART)
    send_sync_packet()
    
    # Start threads
    recorder_thread = threading.Thread(target=recorder, daemon=True, name="Recorder")
    analyzer_thread = threading.Thread(target=analyzer, daemon=True, name="Analyzer")
    
    recorder_thread.start()
    analyzer_thread.start()
    
    print("✅ System started!")
    print("🎙️  Recording audio and analyzing...")
    print("📊 FFT data will be exported based on configuration")
    logging.info("System started")
    print("Press Ctrl+C to stop\n")
    
    try:
        while True:
            time.sleep(300)  # 5 minutes
            send_sync_packet()
            
            # Periodic status update
            current_time = time.strftime("%H:%M:%S")
            print(f"⏰ {current_time} - System running, Queue: {analyze_queue.qsize()}")
            
    except KeyboardInterrupt:
        print("\n🛑 Stopping system...")
        logging.info("Stopping system")
        if ser:
            ser.close()
        print("✅ System stopped")
        logging.info("System stopped")
        
        # Final summary
        try:
            export_files = [f for f in os.listdir(FFT_EXPORT_DIR) if f.endswith('.json')]
            print(f"\n📊 Final Summary: {len(export_files)} FFT exports created")
        except:
            pass

if __name__ == "__main__":
    main()