# coding=utf-8
import logging
import os
import random
import numpy as np
import pandas as pd
import torch.utils.data as da
from pyproj import Transformer
import math
from tqdm import tqdm
import pickle
import hashlib
import json

RNG_SEED = 123
logger = logging.getLogger(__name__)


class DataGenerator(da.Dataset):
    def __init__(self, configs):
        self.configs = configs
        self.rng = random.Random(RNG_SEED)
        self.data_num = 0
        self.datas = []
        
        # Initialize coordinate transformer from WGS84 to EPSG:2180 (Polish coordinate system)
        self.transformer = Transformer.from_crs("EPSG:4326", "EPSG:2180", always_xy=True)

        self.target_size = {'lon': self.configs.delta_lon_size,
                            'lat': self.configs.delta_lat_size,
                            'alt': self.configs.delta_alt_size,
                            'spdx': self.configs.delta_spdx_size,
                            'spdy': self.configs.delta_spdy_size,
                            'spdz': self.configs.delta_spdz_size}

    def _get_cache_key(self, data_path):
        """Generate a cache key based on data path and configuration parameters"""
        # Include relevant config parameters that affect data processing
        config_params = {
            'inp_seq_len': self.configs.inp_seq_len,
            'horizon': self.configs.horizon,
            'data_period': self.configs.data_period,
            'sliding_window_step': getattr(self.configs, 'sliding_window_step', self.configs.inp_seq_len),
        }
        
        # Get list of parquet files and their modification times
        file_info = []
        for root, _, files in os.walk(data_path):
            for f in files:
                if f.endswith('.parquet'):
                    file_path = os.path.join(root, f)
                    try:
                        mtime = os.path.getmtime(file_path)
                        file_info.append((f, mtime))
                    except OSError:
                        continue
        
        # Sort for consistent hashing
        file_info.sort()
        
        # Create hash from config params, data path, and file info
        cache_data = {
            'data_path': data_path,
            'config_params': config_params,
            'file_info': file_info
        }
        
        cache_str = json.dumps(cache_data, sort_keys=True)
        cache_hash = hashlib.md5(cache_str.encode()).hexdigest()
        
        return f"dataset_cache_{cache_hash}.pkl"
    
    def _get_cache_dir(self):
        """Get or create cache directory"""
        cache_dir = os.environ.get('FLIGHTBERT_CACHE_DIR', './cache')
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
        return cache_dir
    
    def _save_dataset_cache(self, data_path):
        """Save processed dataset to cache"""
        try:
            cache_dir = self._get_cache_dir()
            cache_key = self._get_cache_key(data_path)
            cache_path = os.path.join(cache_dir, cache_key)
            
            cache_data = {
                'datas': self.datas,
                'data_num': self.data_num,
                'data_path': data_path,
                'config_params': {
                    'inp_seq_len': self.configs.inp_seq_len,
                    'horizon': self.configs.horizon,
                    'data_period': self.configs.data_period,
                    'sliding_window_step': getattr(self.configs, 'sliding_window_step', self.configs.inp_seq_len),
                }
            }
            
            print(f"Saving dataset cache to: {cache_path}")
            with open(cache_path, 'wb') as f:
                pickle.dump(cache_data, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            print(f"✓ Dataset cache saved successfully ({len(self.datas)} sequences)")
            
        except Exception as e:
            print(f"Warning: Failed to save dataset cache: {e}")
    
    def _load_dataset_cache(self, data_path):
        """Load processed dataset from cache if available and valid"""
        try:
            cache_dir = self._get_cache_dir()
            cache_key = self._get_cache_key(data_path)
            cache_path = os.path.join(cache_dir, cache_key)
            
            if not os.path.exists(cache_path):
                return False
            
            print(f"Found dataset cache: {cache_path}")
            print("Loading cached dataset...")
            
            with open(cache_path, 'rb') as f:
                cache_data = pickle.load(f)
            
            # Verify cache data structure
            if not all(key in cache_data for key in ['datas', 'data_num', 'data_path', 'config_params']):
                print("Warning: Invalid cache data structure, will rebuild dataset")
                return False
            
            # Load cached data
            self.datas = cache_data['datas']
            self.data_num = cache_data['data_num']
            
            print(f"✓ Dataset loaded from cache:")
            print(f"  - Data path: {cache_data['data_path']}")
            print(f"  - Total sequences: {self.data_num}")
            
            # Shuffle if training
            if self.configs.is_training:
                print("Shuffling training data...")
                random.shuffle(self.datas)
                self.data_num = len(self.datas)
            
            return True
            
        except Exception as e:
            print(f"Warning: Failed to load dataset cache: {e}")
            print("Will rebuild dataset from scratch")
            return False

    def load_data_from_dir(self, data_path):
        """Load data from directory with caching support"""
        print(f"Loading data from: {data_path}")
        
        # Try to load from cache first
        if self._load_dataset_cache(data_path):
            return
        
        print("Cache not found or invalid, processing parquet files...")
        
        # First, collect all parquet files
        parquet_files = []
        for root, _, files in os.walk(data_path):
            for f in files:
                if f.endswith('.parquet'):
                    parquet_files.append(os.path.join(root, f))
        
        if len(parquet_files) == 0:
            print(f"Warning: No parquet files found in {data_path}")
            return
        
        print(f"Found {len(parquet_files)} parquet files to process")
        
        # Process files with progress bar
        total_sequences_added = 0
        for parquet_path in tqdm(parquet_files, desc="Processing parquet files", unit="file"):
            try:
                df = pd.read_parquet(parquet_path)
                
                # Validate required columns
                required_columns = ['latitude', 'longitude', 'altitude', 'track', 'groundspeed', 'vertical_rate']
                if not all(col in df.columns for col in required_columns):
                    tqdm.write(f"Skipping {os.path.basename(parquet_path)}: missing required columns")
                    continue
                
                # Sort by timestamp if available (assuming there's a timestamp column)
                if 'timestamp' in df.columns:
                    df = df.sort_values('timestamp')
                
                # Convert to list of records for sliding window processing
                records = []
                for _, row in df.iterrows():
                    # Create record in format compatible with existing processing
                    # Format: timestamp|unused|unused|unused|lon|lat|alt|track|groundspeed|vertical_rate
                    timestamp = row.get('timestamp', 0)  # Use 0 if no timestamp
                    record = f"{timestamp}|0|0|0|{row['longitude']}|{row['latitude']}|{row['altitude']}|{row['track']}|{row['groundspeed']}|{row['vertical_rate']}\n"
                    records.append(record)
                
                sliding_window_step = getattr(self.configs, 'sliding_window_step', self.configs.inp_seq_len)
                sequences_before = len(self.datas)

                if self.configs.data_period == 1:
                    # Process all records as a sliding window
                    while len(records) > self.configs.inp_seq_len + self.configs.horizon:
                        self.datas.append(records[:self.configs.inp_seq_len + self.configs.horizon])
                        records = records[sliding_window_step:]
                else:
                    for i in range(1, self.configs.data_period - 1):
                        sub_records = records[self.configs.data_period - i::self.configs.data_period]
                        while len(sub_records) > self.configs.inp_seq_len + self.configs.horizon:
                            self.datas.append(sub_records[:self.configs.inp_seq_len + self.configs.horizon])
                            sub_records = sub_records[sliding_window_step:]
                
                sequences_added = len(self.datas) - sequences_before
                total_sequences_added += sequences_added
                tqdm.write(f"✓ {os.path.basename(parquet_path)}: {len(df)} records → {sequences_added} sequences")
            
            except Exception as e:
                tqdm.write(f"✗ Error processing {os.path.basename(parquet_path)}: {e}")
                continue

        print(f"\nData processing complete:")
        print(f"- Total files processed: {len(parquet_files)}")
        print(f"- Total sequences created: {len(self.datas)}")
        print(f"- Data path: {data_path}")

        # Save to cache for future use
        if len(self.datas) > 0:
            self._save_dataset_cache(data_path)

        if self.configs.is_training:
            print("Shuffling training data...")
            random.shuffle(self.datas)

        self.data_num = len(self.datas)

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, index):
        seq = self.datas[index]
        return seq

    def convert_lon2binary(self, lon):
        lon = round(lon, 3) * 1000
        bin_lon = '{0:b}'.format(int(lon)).zfill(self.configs.lon_size)
        bin_lon_list = [int(i) for i in bin_lon]
        assert len(bin_lon_list) == self.configs.lon_size, "ERROR"
        return np.array(bin_lon_list)

    def convert_lat2binary(self, lat):
        lat = round(lat, 3) * 1000
        bin_lat = '{0:b}'.format(int(lat)).zfill(self.configs.lat_size)
        bin_lat_list = [int(i) for i in bin_lat]
        assert len(bin_lat_list) == self.configs.lat_size, "ERROR"
        return np.array(bin_lat_list)

    def convert_alt2binary(self, alt):
        alt = int(alt / 10)
        if alt > 15000:
            alt = 0
        bin_alt = '{0:b}'.format(int(alt)).zfill(self.configs.alt_size)
        bin_alt_list = [int(i) for i in bin_alt]

        assert len(bin_alt_list) == self.configs.alt_size, "ERROR {},{}".format(alt, bin_alt)
        return np.array(bin_alt_list)

    def convert_spdx2binary(self, spdx):
        spdx = round(spdx)
        bin_spdx = str(bin(int(spdx)))
        if bin_spdx.startswith('-0b'):
            bin_spdx = '{0:b}'.format(int(-spdx)).zfill(self.configs.spdx_size - 1)
            bin_spdx_list = [1] + [int(i) for i in bin_spdx]
        else:
            bin_spdx = '{0:b}'.format(int(spdx)).zfill(self.configs.spdx_size)
            bin_spdx_list = [int(i) for i in bin_spdx]

        assert len(bin_spdx_list) == self.configs.spdx_size, "ERROR"
        return np.array(bin_spdx_list)

    def convert_spdy2binary(self, spdy):
        spdy = round(spdy)
        bin_spdy = str(bin(int(spdy)))
        if bin_spdy.startswith('-0b'):
            bin_spdy = '{0:b}'.format(int(-spdy)).zfill(self.configs.spdy_size - 1)
            bin_spdy_list = [1] + [int(i) for i in bin_spdy]
        else:
            bin_spdy = '{0:b}'.format(int(spdy)).zfill(self.configs.spdy_size)
            bin_spdy_list = [int(i) for i in bin_spdy]
        assert len(bin_spdy_list) == self.configs.spdy_size, "ERROR"
        return np.array(bin_spdy_list)

    def convert_spdz2binary(self, spdz):
        spdz = round(spdz)
        bin_spdz = bin(int(spdz))
        if bin_spdz.startswith('-0b'):
            bin_spdz = '{0:b}'.format(int(-spdz)).zfill(self.configs.spdz_size - 1)
            bin_spdz_list = [1] + [int(i) for i in bin_spdz]  
        else:
            bin_spdz = '{0:b}'.format(int(spdz)).zfill(self.configs.spdz_size)
            bin_spdz_list = [int(i) for i in bin_spdz]

        assert len(bin_spdz_list) == self.configs.spdz_size, "ERROR"
        return np.array(bin_spdz_list)

    def prepare_minibatch(self, seqs):
        batch_lon, batch_lat, batch_alt, batch_spdx, batch_spdy, batch_spdz = [], [], [], [], [], []
        raw_batch_lon, raw_batch_lat, raw_batch_alt, raw_batch_spdx, raw_batch_spdy, raw_batch_spdz = [], [], [], [], [], []
        batch_t_lon, batch_t_lat, batch_t_alt, batch_t_spdx, batch_t_spdy, batch_t_spdz = [], [], [], [], [], []
        batch_dec_lon, batch_dec_lat, batch_dec_alt, batch_dec_spdx, batch_dec_spdy, batch_dec_spdz = [], [], [], [], [], []

        for seq in seqs:
            seq_lon, seq_lat, seq_alt, seq_spdx, seq_spdy, seq_spdz = [], [], [], [], [], []
            raw_seq_lon, raw_seq_lat, raw_seq_alt, raw_seq_spdx, raw_seq_spdy, raw_seq_spdz = [], [], [], [], [], []
            t_lon, t_lat, t_alt, t_spdx, t_spdy, t_spdz = [], [], [], [], [], []

            for record in seq:
                items = record.strip().split("|")
                # New format: timestamp|unused|unused|unused|lon|lat|alt|track|groundspeed|vertical_rate
                data_time = items[0]
                lon = float(items[4])
                lat = float(items[5]) 
                alt = float(items[6])
                track = float(items[7])
                groundspeed = float(items[8])
                vertical_rate = float(items[9])
                
                # Calculate speed components from track, groundspeed, and vertical_rate
                spdx, spdy, spdz = self.calculate_speed_components(lon, lat, track, groundspeed, vertical_rate)

                seq_lon.append(self.convert_lon2binary(lon))
                seq_lat.append(self.convert_lat2binary(lat))
                seq_alt.append(self.convert_alt2binary(alt))
                seq_spdx.append(self.convert_spdx2binary(spdx))
                seq_spdy.append(self.convert_spdy2binary(spdy))
                seq_spdz.append(self.convert_spdz2binary(spdz))

                raw_seq_lon.append(int(lon * 1000))
                raw_seq_lat.append(int(lat * 1000))
                raw_seq_alt.append(alt // 10)
                raw_seq_spdx.append(int(spdx))
                raw_seq_spdy.append(int(spdy))
                raw_seq_spdz.append(int(spdz))

            for step in range(1, self.configs.inp_seq_len + self.configs.horizon):
                t_lon.append(self.convert_tar((raw_seq_lon[step], raw_seq_lon[step - 1]), 'lon'))
                t_lat.append(self.convert_tar((raw_seq_lat[step], raw_seq_lat[step - 1]), 'lat'))
                t_alt.append(self.convert_tar((raw_seq_alt[step], raw_seq_alt[step - 1]), 'alt'))
                t_spdx.append(self.convert_tar((raw_seq_spdx[step], raw_seq_spdx[step - 1]), 'spdx'))
                t_spdy.append(self.convert_tar((raw_seq_spdy[step], raw_seq_spdy[step - 1]), 'spdy'))
                t_spdz.append(self.convert_tar((raw_seq_spdz[step], raw_seq_spdz[step - 1]), 'spdz'))

            batch_lon.append(seq_lon[:self.configs.inp_seq_len])
            batch_lat.append(seq_lat[:self.configs.inp_seq_len])
            batch_alt.append(seq_alt[:self.configs.inp_seq_len])
            batch_spdx.append(seq_spdx[:self.configs.inp_seq_len])
            batch_spdy.append(seq_spdy[:self.configs.inp_seq_len])
            batch_spdz.append(seq_spdz[:self.configs.inp_seq_len])

            batch_t_lon.append(t_lon[self.configs.inp_seq_len - 1:])
            batch_t_lat.append(t_lat[self.configs.inp_seq_len - 1:])
            batch_t_alt.append(t_alt[self.configs.inp_seq_len - 1:])
            batch_t_spdx.append(t_spdx[self.configs.inp_seq_len - 1:])
            batch_t_spdy.append(t_spdy[self.configs.inp_seq_len - 1:])
            batch_t_spdz.append(t_spdz[self.configs.inp_seq_len - 1:])

            batch_dec_lon.append(t_lon[:self.configs.inp_seq_len - 1])
            batch_dec_lat.append(t_lat[:self.configs.inp_seq_len - 1])
            batch_dec_alt.append(t_alt[:self.configs.inp_seq_len - 1])
            batch_dec_spdx.append(t_spdx[:self.configs.inp_seq_len - 1])
            batch_dec_spdy.append(t_spdy[:self.configs.inp_seq_len - 1])
            batch_dec_spdz.append(t_spdz[:self.configs.inp_seq_len - 1])

            raw_batch_lon.append(raw_seq_lon)
            raw_batch_lat.append(raw_seq_lat)
            raw_batch_alt.append(raw_seq_alt)
            raw_batch_spdx.append(raw_seq_spdx)
            raw_batch_spdy.append(raw_seq_spdy)
            raw_batch_spdz.append(raw_seq_spdz)

        lons = batch_lon
        lats = batch_lat
        alts = batch_alt
        spdxs = batch_spdx
        spdys = batch_spdy
        spdzs = batch_spdz

        return {
            'lon': lons,
            'lat': lats,
            'alt': alts,
            'spdx': spdxs,
            'spdy': spdys,
            'spdz': spdzs,
            'raw_lon': raw_batch_lon,
            'raw_lat': raw_batch_lat,
            'raw_alt': raw_batch_alt,
            'raw_spdx': raw_batch_spdx,
            'raw_spdy': raw_batch_spdy,
            'raw_spdz': raw_batch_spdz,
            't_lon': batch_t_lon,
            't_lat': batch_t_lat,
            't_alt': batch_t_alt,
            't_spdx': batch_t_spdx,
            't_spdy': batch_t_spdy,
            't_spdz': batch_t_spdz,
            'dec_lon': batch_dec_lon,
            'dec_lat': batch_dec_lat,
            'dec_alt': batch_dec_alt,
            'dec_spdx': batch_dec_spdx,
            'dec_spdy': batch_dec_spdy,
            'dec_spdz': batch_dec_spdz,
        }

    def convert_tar(self, d, type='lon'):
        v = d[0] - d[1]

        if v >= 0:
            sign = '0'
        else:
            sign = '1'
            v = -v

        bin = '{0:b}'.format(int(v)).zfill(self.target_size[type] - 1)
        bin = sign + bin
        bin_list = [int(i) for i in bin]

        if len(bin) > self.target_size[type]:
            bin_list = [0] * (self.target_size[type])

        return np.array(bin_list)

    def calculate_speed_components(self, lon, lat, track, groundspeed, vertical_rate):
        """
        Calculate spdx, spdy, spdz from longitude, latitude, track, groundspeed, and vertical_rate
        using coordinate transformation to EPSG:2180
        
        Args:
            lon: longitude in degrees
            lat: latitude in degrees  
            track: track angle in degrees (direction of movement)
            groundspeed: ground speed in m/s or knots (will be normalized)
            vertical_rate: vertical rate in m/s or ft/min (will be normalized)
        
        Returns:
            tuple: (spdx, spdy, spdz) - speed components in m/s
        """
        # Convert track angle to radians
        track_rad = math.radians(track)
        
        # Calculate horizontal speed components
        # spdx = eastward component (positive = east)
        # spdy = northward component (positive = north)
        spdx = groundspeed * math.sin(track_rad)  # East component
        spdy = groundspeed * math.cos(track_rad)  # North component
        
        # spdz is the vertical rate (already available)
        spdz = vertical_rate
        
        return spdx, spdy, spdz

    def clear_cache(self, data_path=None):
        """Clear dataset cache for specific data path or all caches"""
        cache_dir = self._get_cache_dir()
        
        if data_path:
            # Clear specific cache
            cache_key = self._get_cache_key(data_path)
            cache_path = os.path.join(cache_dir, cache_key)
            if os.path.exists(cache_path):
                os.remove(cache_path)
                print(f"✓ Cleared cache for: {data_path}")
            else:
                print(f"No cache found for: {data_path}")
        else:
            # Clear all caches
            if os.path.exists(cache_dir):
                cache_files = [f for f in os.listdir(cache_dir) if f.startswith('dataset_cache_') and f.endswith('.pkl')]
                for cache_file in cache_files:
                    os.remove(os.path.join(cache_dir, cache_file))
                print(f"✓ Cleared {len(cache_files)} cache files")
            else:
                print("No cache directory found")
    
    def get_cache_info(self, data_path):
        """Get information about cache for specific data path"""
        cache_dir = self._get_cache_dir()
        cache_key = self._get_cache_key(data_path)
        cache_path = os.path.join(cache_dir, cache_key)
        
        if os.path.exists(cache_path):
            stat = os.stat(cache_path)
            size_mb = stat.st_size / (1024 * 1024)
            mtime = stat.st_mtime
            
            print(f"Cache info for: {data_path}")
            print(f"  - Cache file: {cache_key}")
            print(f"  - Size: {size_mb:.2f} MB")
            print(f"  - Modified: {pd.to_datetime(mtime, unit='s')}")
            return True
        else:
            print(f"No cache found for: {data_path}")
            return False
