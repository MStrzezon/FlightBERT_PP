import os
import sys
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from traffic.core import Traffic

# Dodaj ścieżkę do projektu
model_dir = os.path.abspath(os.path.dirname(__file__))
sys.path.append(model_dir)

from run import load_torch_model
from model import FlightBERT_PP
from utils import load_config_from_json

# Wczytaj konfigurację i model
config = load_config_from_json('config.json')
model = FlightBERT_PP(config)
load_torch_model(model, None, 'check_points/2025-08-29/epoch_9_0.38805803571428565.pt')
model.eval()

# Wczytaj dane
traffic = Traffic.from_file('traffic_resampled_4s.parquet')

def calculate_speed_components(lon, lat, track, groundspeed, vertical_rate):
    import math
    track_rad = math.radians(track)
    spdx = groundspeed * math.sin(track_rad)
    spdy = groundspeed * math.cos(track_rad)
    spdz = vertical_rate
    return spdx, spdy, spdz

# Użyj funkcji binarnej konwersji z utils.py jeśli są dostępne, w przeciwnym razie zdefiniuj lokalnie
def convert_lon2binary(lon):
    lon = round(lon, 3) * 1000
    bin_lon = '{0:b}'.format(int(lon)).zfill(config.lon_size)
    bin_lon_list = [int(i) for i in bin_lon]
    return np.array(bin_lon_list)

def convert_lat2binary(lat):
    lat = round(lat, 3) * 1000
    bin_lat = '{0:b}'.format(int(lat)).zfill(config.lat_size)
    bin_lat_list = [int(i) for i in bin_lat]
    return np.array(bin_lat_list)

def convert_alt2binary(alt):
    alt = int(alt / 10)
    if alt > 15000:
        alt = 0
    bin_alt = '{0:b}'.format(int(alt)).zfill(config.alt_size)
    bin_alt_list = [int(i) for i in bin_alt]
    return np.array(bin_alt_list)

def convert_spdx2binary(spdx):
    spdx = round(spdx)
    bin_spdx = str(bin(int(spdx)))
    if bin_spdx.startswith('-0b'):
        bin_spdx = '{0:b}'.format(int(-spdx)).zfill(config.spdx_size - 1)
        bin_spdx_list = [1] + [int(i) for i in bin_spdx]
    else:
        bin_spdx = '{0:b}'.format(int(spdx)).zfill(config.spdx_size)
        bin_spdx_list = [int(i) for i in bin_spdx]
    return np.array(bin_spdx_list)

def convert_spdy2binary(spdy):
    spdy = round(spdy)
    bin_spdy = str(bin(int(spdy)))
    if bin_spdy.startswith('-0b'):
        bin_spdy = '{0:b}'.format(int(-spdy)).zfill(config.spdy_size - 1)
        bin_spdy_list = [1] + [int(i) for i in bin_spdy]
    else:
        bin_spdy = '{0:b}'.format(int(spdy)).zfill(config.spdy_size)
        bin_spdy_list = [int(i) for i in bin_spdy]
    return np.array(bin_spdy_list)

def convert_spdz2binary(spdz):
    spdz = round(spdz)
    bin_spdz = str(bin(int(spdz)))
    if bin_spdz.startswith('-0b'):
        bin_spdz = '{0:b}'.format(int(-spdz)).zfill(config.spdz_size - 1)
        bin_spdz_list = [1] + [int(i) for i in bin_spdz]
    else:
        bin_spdz = '{0:b}'.format(int(spdz)).zfill(config.spdz_size)
        bin_spdz_list = [int(i) for i in bin_spdz]
    return np.array(bin_spdz_list)

# Przetwarzanie lotów
flights = [flight for flight in tqdm(traffic)]
all_embeddings = []
labels = []
flight_segments = []

for i, flight in enumerate(tqdm(flights)):
    df = flight.data.reset_index(drop=True)
    segment_len = config.inp_seq_len
    step = config.inp_seq_len
    try:
        for start in range(0, len(df) - segment_len + 1, step):
            df_segment = df.iloc[start:start+segment_len].reset_index(drop=True)
            flight_segments.append(df_segment)
            lon_bin, lat_bin, alt_bin, spdx_bin, spdy_bin, spdz_bin = [], [], [], [], [], []
            for _, row in df_segment.iterrows():
                lon = float(row['longitude'])
                lat = float(row['latitude'])
                alt = float(row['altitude'])
                track = float(row['track'])
                groundspeed = float(row['groundspeed'])
                vertical_rate = float(row['vertical_rate'])
                spdx, spdy, spdz = calculate_speed_components(lon, lat, track, groundspeed, vertical_rate)
                lon_bin.append(convert_lon2binary(lon))
                lat_bin.append(convert_lat2binary(lat))
                alt_bin.append(convert_alt2binary(alt))
                spdx_bin.append(convert_spdx2binary(spdx))
                spdy_bin.append(convert_spdy2binary(spdy))
                spdz_bin.append(convert_spdz2binary(spdz))
            lon_inp = torch.tensor(np.stack(lon_bin)).unsqueeze(0).float()
            lat_inp = torch.tensor(np.stack(lat_bin)).unsqueeze(0).float()
            alt_inp = torch.tensor(np.stack(alt_bin)).unsqueeze(0).float()
            spdx_inp = torch.tensor(np.stack(spdx_bin)).unsqueeze(0).float()
            spdy_inp = torch.tensor(np.stack(spdy_bin)).unsqueeze(0).float()
            spdz_inp = torch.tensor(np.stack(spdz_bin)).unsqueeze(0).float()
            with torch.no_grad():
                cat_inputs = torch.cat((lon_inp, lat_inp, alt_inp, spdx_inp, spdy_inp, spdz_inp), dim=-1)
                token_embeddings = model.enc_embed(cat_inputs)
                token_embeddings = model.pos_emb(token_embeddings.transpose(1, 0)).transpose(1, 0)
                encoder_output = model.transformer_encoder(token_embeddings)
                context_embed = model.feature_aggre(encoder_output)
                all_embeddings.append(context_embed.cpu().numpy().flatten())
                labels.append(i)
    except Exception as e:
        print(f"Error processing flight {flight.callsign} segment starting at index {start}: {e}")
        continue

# Zapisz embeddingi
flight_embeddings_df = pd.DataFrame(all_embeddings)
flight_embeddings_df.to_parquet(os.path.join(traffic_folder, 'flightbertpp_embeddings.parquet'), index=False)
print("Flight embeddings saved to flightbertpp_embeddings.parquet")