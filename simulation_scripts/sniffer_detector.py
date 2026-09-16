"""
=============================================================================================================
S7COMM & MODBUS REAL-TIME ML ANOMALY DETECTOR (REDIS DECOUPLED PIPELINE)
=============================================================================================================
Process 1 (Producer): Sniffs S7comm (port 102) & Modbus TCP (port 502) using Scapy using switch span or port 
                    mirroring, extracts features, and publishes telemetry payloads directly to Redis Pub/Sub.

Process 2 (Consumer): Subscribes to Redis stream, runs ML model inference, and dispatches incident
                     summaries to Ollama LLM and Telegram.

Supported Models:
- Isolation Forest (Unsupervised - Fast)
- XGBoost (Supervised - Accurate)
- LSTM-Autoencoder (Deep Learning - Temporal patterns)
=============================================================================================================
=============================================== Hoe to use it ?? 

python sniffer_detector.py --model lstm_autoencoder (isolation_forest or xgboost)

# Use LSTM with custom threshold
python sniffer_detector.py --model lstm_autoencoder --threshold 2.0

# Test mode with LSTM
python sniffer_detector.py --test --pcap "../Datasets_wireshark/s7comm_cap.pcapng" --model xgboost

# Ensemble (all models)
python sniffer_detector.py --model ensemble

# run heuristic only if the models seem failing 
python sniffer_detector.py 
"""

import json
import multiprocessing
import struct
import joblib
import numpy as np
import redis
import requests
import warnings
import os
import sys
import torch
import torch.nn as nn
from collections import deque
from datetime import datetime
from scapy.all import IP, TCP, Raw, sniff
import threading
import time
from collections import deque
from datetime import datetime, timedelta

warnings.filterwarnings('ignore')

# ==========================================
# 1. CONFIGURATION
# ==========================================

# Redis/Memurai Configuration
REDIS_HOST = "localhost"
REDIS_PORT = 6379
REDIS_CHANNEL = "ics_telemetry_stream"

# Model Configuration
MODEL_DIR = "../outputs"
ISOLATION_FOREST_PATH = os.path.join(MODEL_DIR, "isolation_forest.pkl")
XGBOOST_PATH = os.path.join(MODEL_DIR, "xgboost_model.pkl")
LSTM_PATH = os.path.join(MODEL_DIR, "lstm_autoencoder.pt")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")
CONFIG_PATH = os.path.join(MODEL_DIR, "model_config.json")

# Model Selection: 'isolation_forest', 'xgboost', 'lstm_autoencoder', or 'ensemble'
MODEL_TYPE = "lstm_autoencoder"  # Change as needed

# LSTM Configuration (must match training config)
LSTM_SEQ_LEN = 10
LSTM_FEATURES = 73
LSTM_LATENT_DIM = 32
LSTM_HIDDEN_DIM = 128
LSTM_NUM_CLASSES = 5
LSTM_THRESHOLD = 1.5  # Anomaly threshold (from training)

# Packet buffering for LSTM
BUFFER_SIZE = LSTM_SEQ_LEN
packet_buffer = deque(maxlen=BUFFER_SIZE)
feature_buffer = deque(maxlen=BUFFER_SIZE)

# Ollama Configuration
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.2"

# Telegram Configuration
TELEGRAM_BOT_TOKEN = "[Redacted]"
TELEGRAM_CHAT_ID = "[Redacted]"

# Ports
S7_PORT = 102
MODBUS_PORT = 502
S7_PROTOCOL_ID = 0x32
S7PLUS_PROTOCOL_ID = 0x72

# =============================================
# 2. FEATURE COLS - MATCHING NOTEBOOK 1
# =============================================

FEATURE_COLS = [
    'packet_length', 'transaction_id', 'protocol_id', 'unit_id', 'function_code',
    'reference_num', 'word_count', 'payload_size', 'sport', 'dport', 'ttl',
    'ip_flags', 'tcp_flags', 'seq', 'ack', 'window_size', 'pdu_type',
    'pdu_reference', 'subfunction', 'area_code', 'pdu_length', 'protocol_version',
    'sequence', 'opcode', 'message_id', 'Length', 'start', 'end', 'startOffset',
    'endOffset', 'duration', 'sPackets', 'rPackets', 'sBytesSum', 'rBytesSum',
    'sBytesMax', 'rBytesMax', 'sBytesMin', 'rBytesMin', 'sBytesAvg', 'rBytesAvg',
    'sLoad', 'rLoad', 'sPayloadSum', 'rPayloadSum', 'sPayloadMax', 'rPayloadMax',
    'sPayloadMin', 'rPayloadMin', 'sPayloadAvg', 'rPayloadAvg', 'sInterPacketAvg',
    'rInterPacketAvg', 'sttl', 'rttl', 'sAckRate', 'rAckRate', 'sFinRate',
    'rFinRate', 'sPshRate', 'rPshRate', 'sSynRate', 'rSynRate', 'sRstRate',
    'rRstRate', 'sWinTCP', 'rWinTCP', 'sAckDelayMax', 'rAckDelayMax',
    'sAckDelayMin', 'rAckDelayMin', 'sAckDelayAvg', 'rAckDelayAvg'
]

# =============================================
# 3. LSTM-AUTOENCODER ARCHITECTURE
# =============================================
class MultiTaskLSTMAutoencoder(nn.Module):
    """
    Multi-Task LSTM Autoencoder matching the trained model from Notebook 2.
    """
    def __init__(self, seq_len, num_features, latent_dim, num_classes, hidden_dim=128):
        super(MultiTaskLSTMAutoencoder, self).__init__()
        self.seq_len = seq_len
        self.num_features = num_features
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        
        # Shared Encoder
        self.encoder_lstm1 = nn.LSTM(num_features, hidden_dim, batch_first=True)
        self.encoder_lstm2 = nn.LSTM(hidden_dim, latent_dim, batch_first=True)
        
        # Decoder Branch (Reconstruction)
        self.decoder_lstm1 = nn.LSTM(latent_dim, hidden_dim, batch_first=True)
        self.decoder_lstm2 = nn.LSTM(hidden_dim, num_features, batch_first=True)
        
        # Classifier Branch (Multi-class Classification)
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        # Shared Encoder
        x, _ = self.encoder_lstm1(x)
        x, (hidden, cell) = self.encoder_lstm2(x)
        
        # Bottleneck (latent representation)
        latent = hidden[-1]
        
        # Branch 1: Reconstruction Decoder
        latent_seq = latent.unsqueeze(1).repeat(1, self.seq_len, 1)
        x_recon, _ = self.decoder_lstm1(latent_seq)
        x_recon, _ = self.decoder_lstm2(x_recon)
        
        # Branch 2: Classifier
        class_logits = self.classifier(latent)
        
        return x_recon, class_logits
    
    def encode(self, x):
        """Extract latent representation for downstream tasks."""
        x, _ = self.encoder_lstm1(x)
        x, (hidden, _) = self.encoder_lstm2(x)
        return hidden[-1]
    
    def get_reconstruction(self, x):
        """Get reconstruction only (for anomaly detection)."""
        recon, _ = self.forward(x)
        return recon
    
    def get_classification(self, x):
        """Get classification logits only."""
        _, logits = self.forward(x)
        return logits


# =============================================
# 4. FEATURE EXTRACTION ENGINE
# =============================================

def extract_s7_features(packet) -> dict:
    """
    Enhanced S7comm/S7comm+ feature extraction matching Notebook 1 parsers.
    """
    if not packet.haslayer(TCP) or not packet.haslayer(Raw):
        return None
    
    payload = bytes(packet[Raw].load)
    pkt_length = len(packet)
    timestamp = packet.time
    
    # Get TCP features
    sport = packet[TCP].sport
    dport = packet[TCP].dport
    tcp_flags = int(packet[TCP].flags)
    seq_num = packet[TCP].seq
    ack_num = packet[TCP].ack
    window_size = packet[TCP].window
    
    # IP features
    src_ip = packet[IP].src
    dst_ip = packet[IP].dst
    ttl = packet[IP].ttl
    ip_flags = int(packet[IP].flags)
    frag_offset = packet[IP].frag
    
    if len(payload) < 14:
        return None
    
    try:
        cotp_len = payload[4]
        s7_offset = 4 + 1 + cotp_len
        
        if len(payload) < s7_offset + 10:
            return None
        
        proto_id = payload[s7_offset]
        if proto_id not in [S7_PROTOCOL_ID, S7PLUS_PROTOCOL_ID]:
            return None
        
        # Parse S7comm header
        pdu_type = payload[s7_offset + 1]
        pdu_length = int.from_bytes(payload[s7_offset + 6:s7_offset + 8], byteorder='big')
        pdu_ref = int.from_bytes(payload[s7_offset + 4:s7_offset + 6], byteorder='big')
        function_code = payload[s7_offset + 8] if len(payload) > s7_offset + 8 else 0
        subfunction = payload[s7_offset + 9] if len(payload) > s7_offset + 9 else 0
        
        area_code = 0
        if len(payload) > s7_offset + 12:
            area_code = payload[s7_offset + 12]
        
        payload_size = max(0, len(payload) - s7_offset)
        
        # S7comm+ specific features
        proto_version = 0
        sequence = 0
        opcode = 0
        message_id = 0
        is_request = 1 if pdu_type in [0x01, 0x03] else 0
        
        if proto_id == S7PLUS_PROTOCOL_ID:
            proto_version = payload[s7_offset + 1] if len(payload) > s7_offset + 1 else 0
            sequence = (payload[s7_offset + 6] << 8) | payload[s7_offset + 7]
            opcode = payload[s7_offset + 8] if len(payload) > s7_offset + 8 else 0
            message_id = payload[s7_offset + 12] if len(payload) > s7_offset + 12 else 0
        
        # Extract telemetry if available (DB1 read/write)
        pression = 0.0
        temp_four = 0.0
        vitesse_conv = 0.0
        has_telemetry = False
        transaction_id = 0
        
        # Look for DB1 data (typically function_code for read/write)
        if function_code in [0x04, 0x05] and payload_size >= 16:
            try:
                if pdu_type == 0x02:  # Response
                    data_start = s7_offset + 10 + 4
                else:  # Request
                    data_start = s7_offset + 10
                
                if len(payload) >= data_start + 16:
                    temp_bytes = payload[data_start:data_start+4]
                    pres_bytes = payload[data_start+4:data_start+8]
                    vit_bytes = payload[data_start+8:data_start+12]
                    
                    temp = struct.unpack('>f', temp_bytes)[0]
                    pres = struct.unpack('>f', pres_bytes)[0]
                    vit = struct.unpack('>f', vit_bytes)[0]
                    
                    # Validate physical bounds
                    if 0.0 <= pres <= 10.0 and 0.0 <= temp <= 60.0:
                        pression = pres
                        temp_four = temp
                        vitesse_conv = vit
                        has_telemetry = True
                        transaction_id = int.from_bytes(payload[s7_offset+2:s7_offset+4], byteorder='big')
            except:
                pass
        
        # Build full feature dictionary
        features = {
            'protocol': 'S7COMM' if proto_id == S7_PROTOCOL_ID else 'S7COMM_PLUS',
            'protocol_id': proto_id,
            'source_protocol': 's7comm' if proto_id == S7_PROTOCOL_ID else 's7comm_plus',
            'pdu_type': pdu_type,
            'pdu_reference': pdu_ref,
            'function_code': function_code,
            'subfunction': subfunction,
            'area_code': area_code,
            'pdu_length': pdu_length,
            'payload_size': payload_size,
            'message_id': message_id,
            'is_request': is_request,
            'protocol_version': proto_version,
            'sequence': sequence,
            'opcode': opcode,
            'packet_length': pkt_length,
            'timestamp': timestamp,
            'sport': sport,
            'dport': dport,
            'ttl': ttl,
            'ip_flags': ip_flags,
            'frag_offset': frag_offset,
            'tcp_flags': tcp_flags,
            'seq': seq_num,
            'ack': ack_num,
            'window_size': window_size,
            'src_ip': src_ip,
            'dst_ip': dst_ip,
            'pression': pression,
            'temp_four': temp_four,
            'vitesse_conv': vitesse_conv,
            'has_telemetry': has_telemetry,
            'transaction_id': transaction_id,
            'reference_num': 0,
            'word_count': 0,
            'unit_id': 0,
            'Length': pkt_length,
            'start': 0,
            'end': 0,
            'startOffset': 0,
            'endOffset': 0,
            'duration': 0.0,
            'sPackets': 1,
            'rPackets': 0,
            'sBytesSum': pkt_length,
            'rBytesSum': 0,
            'sBytesMax': pkt_length,
            'rBytesMax': 0,
            'sBytesMin': pkt_length,
            'rBytesMin': 0,
            'sBytesAvg': float(pkt_length),
            'rBytesAvg': 0.0,
            'sLoad': 0.0,
            'rLoad': 0.0,
            'sPayloadSum': payload_size,
            'rPayloadSum': 0,
            'sPayloadMax': payload_size,
            'rPayloadMax': 0,
            'sPayloadMin': payload_size,
            'rPayloadMin': 0,
            'sPayloadAvg': float(payload_size),
            'rPayloadAvg': 0.0,
            'sInterPacketAvg': 0.0,
            'rInterPacketAvg': 0.0,
            'sttl': ttl,
            'rttl': 0,
            'sAckRate': 1.0 if tcp_flags & 0x10 else 0.0,
            'rAckRate': 0.0,
            'sFinRate': 1.0 if tcp_flags & 0x01 else 0.0,
            'rFinRate': 0.0,
            'sPshRate': 1.0 if tcp_flags & 0x08 else 0.0,
            'rPshRate': 0.0,
            'sSynRate': 1.0 if tcp_flags & 0x02 else 0.0,
            'rSynRate': 0.0,
            'sRstRate': 1.0 if tcp_flags & 0x04 else 0.0,
            'rRstRate': 0.0,
            'sWinTCP': window_size,
            'rWinTCP': 0,
            'sAckDelayMax': 0.0,
            'rAckDelayMax': 0.0,
            'sAckDelayMin': 0.0,
            'rAckDelayMin': 0.0,
            'sAckDelayAvg': 0.0,
            'rAckDelayAvg': 0.0,
            'label': 0
        }
        
        return features
        
    except Exception as e:
        return None


def extract_modbus_features(packet) -> dict:
    """Enhanced Modbus TCP feature extraction matching Notebook 1 parsers."""
    if not packet.haslayer(TCP) or not packet.haslayer(Raw):
        return None
    
    payload = bytes(packet[Raw].load)
    pkt_length = len(packet)
    timestamp = packet.time
    
    sport = packet[TCP].sport
    dport = packet[TCP].dport
    tcp_flags = int(packet[TCP].flags)
    seq_num = packet[TCP].seq
    ack_num = packet[TCP].ack
    window_size = packet[TCP].window
    
    src_ip = packet[IP].src
    dst_ip = packet[IP].dst
    ttl = packet[IP].ttl
    ip_flags = int(packet[IP].flags)
    frag_offset = packet[IP].frag
    
    if len(payload) < 8:
        return None
    
    try:
        trans_id = int.from_bytes(payload[0:2], byteorder='big')
        proto_id = int.from_bytes(payload[2:4], byteorder='big')
        length = int.from_bytes(payload[4:6], byteorder='big')
        unit_id = payload[6]
        func_code = payload[7]
        
        if proto_id != 0:
            return None
        
        ref_num = 0
        word_count = 1
        
        if len(payload) >= 10:
            ref_num = int.from_bytes(payload[8:10], byteorder='big')
        if len(payload) >= 12:
            word_count = int.from_bytes(payload[10:12], byteorder='big')
        
        payload_size = max(0, len(payload) - 8)
        
        pression = 0.0
        temp_four = 0.0
        vitesse_conv = 0.0
        has_telemetry = False
        
        if func_code in [0x03, 0x04] and len(payload) >= 13:
            byte_count = payload[8]
            data_start = 9
            if len(payload) >= data_start + 12:
                try:
                    temp_bytes = payload[data_start:data_start+4]
                    pres_bytes = payload[data_start+4:data_start+8]
                    vit_bytes = payload[data_start+8:data_start+12]
                    
                    temp = struct.unpack('>f', temp_bytes)[0]
                    pres = struct.unpack('>f', pres_bytes)[0]
                    vit = struct.unpack('>f', vit_bytes)[0]
                    
                    if 0.0 <= pres <= 10.0 and 0.0 <= temp <= 60.0:
                        pression = pres
                        temp_four = temp
                        vitesse_conv = vit
                        has_telemetry = True
                except:
                    pass
        
        features = {
            'protocol': 'MODBUS_TCP',
            'protocol_id': proto_id,
            'source_protocol': 'modbus',
            'transaction_id': trans_id,
            'unit_id': unit_id,
            'function_code': func_code,
            'reference_num': ref_num,
            'word_count': word_count,
            'payload_size': payload_size,
            'packet_length': pkt_length,
            'timestamp': timestamp,
            'sport': sport,
            'dport': dport,
            'ttl': ttl,
            'ip_flags': ip_flags,
            'frag_offset': frag_offset,
            'tcp_flags': tcp_flags,
            'seq': seq_num,
            'ack': ack_num,
            'window_size': window_size,
            'src_ip': src_ip,
            'dst_ip': dst_ip,
            'pression': pression,
            'temp_four': temp_four,
            'vitesse_conv': vitesse_conv,
            'has_telemetry': has_telemetry,
            'pdu_type': unit_id,
            'pdu_reference': 0,
            'subfunction': 0,
            'area_code': 0,
            'pdu_length': length,
            'protocol_version': 0,
            'sequence': 0,
            'opcode': 0,
            'message_id': 0,
            'is_request': 1 if func_code in [0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x0F, 0x10] else 0,
            'Length': pkt_length,
            'start': 0,
            'end': 0,
            'startOffset': 0,
            'endOffset': 0,
            'duration': 0.0,
            'sPackets': 1,
            'rPackets': 0,
            'sBytesSum': pkt_length,
            'rBytesSum': 0,
            'sBytesMax': pkt_length,
            'rBytesMax': 0,
            'sBytesMin': pkt_length,
            'rBytesMin': 0,
            'sBytesAvg': float(pkt_length),
            'rBytesAvg': 0.0,
            'sLoad': 0.0,
            'rLoad': 0.0,
            'sPayloadSum': payload_size,
            'rPayloadSum': 0,
            'sPayloadMax': payload_size,
            'rPayloadMax': 0,
            'sPayloadMin': payload_size,
            'rPayloadMin': 0,
            'sPayloadAvg': float(payload_size),
            'rPayloadAvg': 0.0,
            'sInterPacketAvg': 0.0,
            'rInterPacketAvg': 0.0,
            'sttl': ttl,
            'rttl': 0,
            'sAckRate': 1.0 if tcp_flags & 0x10 else 0.0,
            'rAckRate': 0.0,
            'sFinRate': 1.0 if tcp_flags & 0x01 else 0.0,
            'rFinRate': 0.0,
            'sPshRate': 1.0 if tcp_flags & 0x08 else 0.0,
            'rPshRate': 0.0,
            'sSynRate': 1.0 if tcp_flags & 0x02 else 0.0,
            'rSynRate': 0.0,
            'sRstRate': 1.0 if tcp_flags & 0x04 else 0.0,
            'rRstRate': 0.0,
            'sWinTCP': window_size,
            'rWinTCP': 0,
            'sAckDelayMax': 0.0,
            'rAckDelayMax': 0.0,
            'sAckDelayMin': 0.0,
            'rAckDelayMin': 0.0,
            'sAckDelayAvg': 0.0,
            'rAckDelayAvg': 0.0,
            'label': 0
        }
        
        return features
        
    except Exception as e:
        return None


# =============================================
# 5. FEATURE VECTOR BUILDER
# =============================================

def build_feature_vector(features_dict: dict) -> np.ndarray:
    """Build complete feature vector matching Notebook 1 training features."""
    feature_values = []
    
    for col in FEATURE_COLS:
        value = features_dict.get(col, 0.0)
        if col == 'pression':
            value = features_dict.get('pression', 0.0)
        elif col == 'temp_four':
            value = features_dict.get('temp_four', 0.0)
        elif col == 'vitesse_conv':
            value = features_dict.get('vitesse_conv', 0.0)
        
        feature_values.append(float(value))
    
    return np.array(feature_values)


# ==========================================
# 6. MODEL LOADER (WITH LSTM SUPPORT)
# ==========================================

class ModelLoader:
    """Load and manage ML models for inference."""
    
    def __init__(self):
        self.model = None
        self.scaler = None
        self.model_type = None
        self.feature_cols = FEATURE_COLS
        self.loaded = False
        self.lstm_model = None
        self.lstm_config = {}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def load_models(self, model_type="isolation_forest"):
        """Load the specified model and scaler."""
        self.model_type = model_type
        
        try:
            # Load scaler
            if os.path.exists(SCALER_PATH):
                self.scaler = joblib.load(SCALER_PATH)
                print(f"[CONSUMER] Loaded scaler from '{SCALER_PATH}'")
            else:
                print("[CONSUMER] No scaler found - using raw features")
            
            # Load model based on type
            if model_type == "isolation_forest":
                if os.path.exists(ISOLATION_FOREST_PATH):
                    self.model = joblib.load(ISOLATION_FOREST_PATH)
                    print(f"[CONSUMER] Loaded Isolation Forest from '{ISOLATION_FOREST_PATH}'")
                else:
                    print(f"[CONSUMER] Isolation Forest not found at '{ISOLATION_FOREST_PATH}'")
                    return False
                    
            elif model_type == "xgboost":
                if os.path.exists(XGBOOST_PATH):
                    self.model = joblib.load(XGBOOST_PATH)
                    print(f"[CONSUMER] Loaded XGBoost from '{XGBOOST_PATH}'")
                else:
                    print(f"[CONSUMER] XGBoost not found at '{XGBOOST_PATH}'")
                    return False
            
            elif model_type == "lstm_autoencoder":
                if self.load_lstm_model():
                    print(f"[CONSUMER] Loaded LSTM-Autoencoder from '{LSTM_PATH}'")
                else:
                    print(f"[CONSUMER] LSTM-Autoencoder not found at '{LSTM_PATH}'")
                    return False
            
            elif model_type == "ensemble":
                # Load all models for ensemble voting
                success = True
                if os.path.exists(ISOLATION_FOREST_PATH):
                    self.model = joblib.load(ISOLATION_FOREST_PATH)
                    print(f"[CONSUMER] Loaded Isolation Forest")
                else:
                    success = False
                
                if os.path.exists(XGBOOST_PATH):
                    self.xgb_model = joblib.load(XGBOOST_PATH)
                    print(f"[CONSUMER] Loaded XGBoost")
                else:
                    success = False
                
                if self.load_lstm_model():
                    print(f"[CONSUMER] Loaded LSTM-Autoencoder")
                else:
                    success = False
                
                if not success:
                    print("[CONSUMER] Ensemble load failed - some models missing")
                    return False
            
            else:
                print(f"[CONSUMER] Unknown model type: {model_type}")
                return False
            
            # Load config if exists
            if os.path.exists(CONFIG_PATH):
                with open(CONFIG_PATH, 'r') as f:
                    self.config = json.load(f)
                print(f"[CONSUMER] Loaded config from '{CONFIG_PATH}'")
                # Update LSTM threshold from config
                if 'threshold' in self.config:
                    LSTM_THRESHOLD = self.config['threshold']
            
            self.loaded = True
            return True
            
        except Exception as e:
            print(f"[!] Error loading models: {e}")
            return False
    
    def load_lstm_model(self):
        """Load LSTM-Autoencoder model."""
        try:
            # Initialize model with correct architecture
            self.lstm_model = MultiTaskLSTMAutoencoder(
                seq_len=LSTM_SEQ_LEN,
                num_features=LSTM_FEATURES,
                latent_dim=LSTM_LATENT_DIM,
                num_classes=LSTM_NUM_CLASSES,
                hidden_dim=LSTM_HIDDEN_DIM
            ).to(self.device)
            
            # Load weights
            if os.path.exists(LSTM_PATH):
                self.lstm_model.load_state_dict(
                    torch.load(LSTM_PATH, map_location=self.device)
                )
                self.lstm_model.eval()
                return True
            return False
        except Exception as e:
            print(f"[!] LSTM load error: {e}")
            return False
    
    def predict_lstm(self, sequence):
        """Run LSTM prediction on a sequence of features."""
        if self.lstm_model is None:
            return None, False
        
        try:
            # Ensure sequence has correct shape
            if len(sequence.shape) == 2:
                sequence = sequence.reshape(1, sequence.shape[0], sequence.shape[1])
            elif len(sequence.shape) == 3:
                sequence = sequence.reshape(1, sequence.shape[1], sequence.shape[2])
            
            # Convert to tensor
            seq_tensor = torch.tensor(sequence, dtype=torch.float32).to(self.device)
            
            with torch.no_grad():
                # Get reconstruction and classification
                recon, class_logits = self.lstm_model(seq_tensor)
                
                # Calculate reconstruction error
                recon_error = torch.mean((recon - seq_tensor) ** 2).item()
                
                # Get classification
                class_pred = torch.argmax(class_logits, dim=1).item()
                
                # Determine anomaly based on reconstruction error
                is_anomaly = recon_error > LSTM_THRESHOLD
                
                return class_pred, is_anomaly, recon_error
                
        except Exception as e:
            print(f"[!] LSTM prediction error: {e}")
            return None, False, 0.0

    def predict(self, feature_vector):
        """Run prediction on a single feature vector (for non-LSTM models)."""
        if not self.loaded:
            return None, False
        
        try:
            # Scale if scaler exists
            if self.scaler is not None:
                feature_vector = self.scaler.transform(feature_vector.reshape(1, -1))
            else:
                feature_vector = feature_vector.reshape(1, -1)
            
            # Run prediction based on model type
            if self.model_type == "isolation_forest":
                prediction = self.model.predict(feature_vector)
                is_anomaly = prediction[0] == -1
                return prediction[0], is_anomaly
                
            elif self.model_type == "xgboost":
                prediction = self.model.predict(feature_vector)
                prediction_label = int(prediction[0])
                is_anomaly = prediction_label != 0
                return prediction_label, is_anomaly
            
            elif self.model_type == "ensemble":
                return self.predict_ensemble(feature_vector)
                
            else:
                return None, False
                
        except Exception as e:
            print(f"[!] Prediction error: {e}")
            return None, False
    
    def predict_ensemble(self, feature_vector):
        """Run ensemble prediction across all models."""

        predictions = []
        anomalies = []
        
        try:   # Isolation Forest
            if self.model is not None:
                pred = self.model.predict(feature_vector)
                predictions.append(0 if pred[0] == 1 else 1)
                anomalies.append(pred[0] == -1)
        except:
            pass
        
        try:   # Xgboost
            if hasattr(self, 'xgb_model') and self.xgb_model is not None:
                pred = self.xgb_model.predict(feature_vector)
                predictions.append(int(pred[0]))
                anomalies.append(int(pred[0]) != 0)
        except:
            pass
        
        # LSTM would need sequence, skip in single-packet mode
        if predictions:
            # Majority vote
            final_pred = max(set(predictions), key=predictions.count)
            is_anomaly = final_pred != 0 or any(anomalies)
            return final_pred, is_anomaly
        
        return None, False


# ==========================================
# 7. HEURISTIC ANOMALY DETECTION
# ==========================================
def check_heuristic_anomaly(features: dict) -> tuple:
    """Heuristic-based anomaly detection with physical bounds."""
    PRESSURE_MIN, PRESSURE_MAX = 0.0, 5.5
    TEMP_MIN, TEMP_MAX = 15.0, 35.0
    SPEED_MIN, SPEED_MAX = 0.0, 2.5
    
    pression = features.get('pression', 0.0)
    temp = features.get('temp_four', 0.0)
    speed = features.get('vitesse_conv', 0.0)
    
    anomalies = []
    
    if pression > PRESSURE_MAX:
        anomalies.append(f"Pression spike: {pression:.2f} bar")
    elif pression < PRESSURE_MIN:
        anomalies.append(f"Pression drop: {pression:.2f} bar")
    
    if temp > TEMP_MAX:
        anomalies.append(f"Température élevée: {temp:.1f}°C")
    elif temp < TEMP_MIN:
        anomalies.append(f"Température basse: {temp:.1f}°C")
    
    if speed > SPEED_MAX:
        anomalies.append(f"Vitesse excessive: {speed:.2f} m/min")
    
    func_code = features.get('function_code', 0)
    if func_code in [0x0F, 0x10]:
        anomalies.append(f"Écriture illégale en mémoire (FC: {hex(func_code)})")
    
    is_anomaly = len(anomalies) > 0
    anomaly_type = " | ".join(anomalies) if is_anomaly else "Normal"
    
    return is_anomaly, anomaly_type


# ==========================================
# 8. OLLAMA SUMMARY 
# ==========================================
def query_ollama_summary(anomaly_data: dict, anomaly_type: str, prediction_label: int = 0) -> str:
    """Génère un résumé opérationnel en français via Ollama."""
    
    attack_types = {
        0: "Normal / Bénin",
        1: "MITM / PLC Attack",
        2: "Port Scan / Reconnaissance",
        3: "Telnet PLC Attack",
        4: "Web Access Attack"
    }
    predicted_attack = attack_types.get(prediction_label, "Inconnu")
    

    # Ce prompt doit etre envoye chaque fois avec l'alerte 
    prompt = f"""
    [ALERTE ICS/SCADA]
    Vous êtes un expert en cybersécurité industrielle. Analysez l'alerte suivante et fournissez un rapport concis pour l'opérateur SOC en réponse concise et opérationnelle.

    Anomalie: {anomaly_type}
    Source: {anomaly_data.get('src_ip', 'Inconnue')}
    Protocole: {anomaly_data.get('protocol', 'Inconnu')}
    Classification ML: {predicted_attack}

    Télémétrie:
    - Pression: {anomaly_data.get('pression', 0):.2f} bar (normal: <5.5)
    - Température: {anomaly_data.get('temp_four', 0):.1f} °C (normal: 15-35°C)
    - Vitesse: {anomaly_data.get('vitesse_conv', 0):.2f} m/min (normal: <2.5)

    Rapport:
    1. Nature: Physique ou Cyber?
    2. Criticité: Faible/Moyen/Critique
    3. Action: Que faire maintenant?
    """
    
    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
                "temperature": 0.3,
                "top_p": 0.9
            },
            timeout=30,
        )
        if response.status_code == 200:
            result = response.json().get("response", "Aucune réponse générée.")
            return result
    except Exception as e:
        print(f"[!] Ollama API Erreur: {e}")

    return "[Erreur] Agent IA indisponible. Inspection manuelle requise."


# ==========================================
# 9. TELEGRAM ALERT
# ==========================================
def send_telegram_alert(summary_text: str, anomaly_data: dict, prediction_label: int = 0):
    """Dispatches human-readable alert to Telegram bot."""
    if TELEGRAM_BOT_TOKEN == "YOUR_TELEGRAM_BOT_TOKEN":
        print("[*] Telegram notification skipped (Token not configured).")
        return

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    
    attack_types = {
        0: "Normal / Bénin",
        1: "MITM / PLC Attack",
        2: "Port Scan / Reconnaissance",
        3: "Telnet PLC Attack",
        4: "Web Access Attack"
    }
    predicted_attack = attack_types.get(prediction_label, "Inconnu")
    
    message = (
        f"🚨 <b>ICS ANOMALY DETECTED</b> 🚨\n"
        f"<b>Protocol:</b> {anomaly_data.get('protocol', 'Unknown')}\n"
        f"<b>Source IP:</b> {anomaly_data.get('src_ip', 'Unknown')}\n"
        f"<b>Destination IP:</b> {anomaly_data.get('dst_ip', 'Unknown')}\n"
        f"<b>Time:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"<b>Classification:</b> {predicted_attack}\n\n"
        f"<b>Telemetry:</b>\n"
        f"• Pression: {anomaly_data.get('pression', 0):.2f} bar\n"
        f"• Température: {anomaly_data.get('temp_four', 0):.1f} °C\n"
        f"• Vitesse: {anomaly_data.get('vitesse_conv', 0):.2f} m/min\n\n"
        f"<b>Agent Analysis:</b>\n{summary_text}"
    )

    try:
        requests.post(
            url,
            json={
                "chat_id": TELEGRAM_CHAT_ID,
                "text": message,
                "parse_mode": "HTML",
            },
            timeout=5,
        )
        print("[+] Alert successfully dispatched to Telegram.")
    except Exception as e:
        print(f"[!] Failed to send Telegram alert: {e}")


# ==========================================
# 10. PACKET BUFFER MANAGER FOR LSTM
# ==========================================

class PacketBuffer:
    """Buffer for collecting packets into sequences for LSTM."""
    
    def __init__(self, buffer_size=LSTM_SEQ_LEN):
        self.buffer_size = buffer_size
        self.packets = deque(maxlen=buffer_size)
        self.features = deque(maxlen=buffer_size)
        self.last_sequence_time = 0
    
    def add_packet(self, features_dict):
        """Add a packet to the buffer."""
        feature_vector = build_feature_vector(features_dict)
        self.packets.append(features_dict)
        self.features.append(feature_vector)
        return len(self.features) >= self.buffer_size
    
    def get_sequence(self):
        """Get the current sequence as a numpy array."""
        if len(self.features) < self.buffer_size:
            return None
        
        sequence = np.array(self.features)
        
        if model_loader.scaler is not None:
            sequence = model_loader.scaler.transform(sequence)
        
        return sequence
    
    def get_telemetry_data(self):
        """Get the last packet's telemetry data(pres, temp, vit) for alerting."""
        if len(self.packets) > 0:
            return self.packets[-1]
        return {}
    
    def clear(self):
        """Clear the buffer."""
        self.packets.clear()
        self.features.clear()


# ==========================================
# 11. PROCESS 1: PRODUCER (SNIFFER TO REDIS)
# ==========================================

def run_sniffer_producer():
    """Captures S7comm and Modbus TCP packets and pushes features to Redis."""
    try:
        redis_ch = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=0)
        redis_ch.ping()
        print(f"[PRODUCER] Connected to Redis on {REDIS_HOST}:{REDIS_PORT}")
    except redis.ConnectionError:
        print(f"[!] Cannot connect to Redis/Memurai server")
        print("    Make sure Memurai is running on your system")
        return

    packet_count = 0
    telemetry_count = 0

    def process_packet(packet):
        nonlocal packet_count, telemetry_count
        packet_count += 1
        
        if packet_count % 100 == 0:
            print(f"[PRODUCER] Processed {packet_count} packets, {telemetry_count} with telemetry")
        
        if not packet.haslayer(TCP) or not packet.haslayer(Raw):
            return

        dport = packet[TCP].dport
        sport = packet[TCP].sport
        features = None

        if dport == S7_PORT or sport == S7_PORT:
            features = extract_s7_features(packet)
        elif dport == MODBUS_PORT or sport == MODBUS_PORT:
            features = extract_modbus_features(packet)

        if not features or not features.get("has_telemetry", False):
            return

        telemetry_count += 1

        try:
            redis_ch.publish(REDIS_CHANNEL, json.dumps(features))
            print(f"[PRODUCER] Published telemetry: {features['protocol']} | "
                  f"P={features['pression']:.2f} | T={features['temp_four']:.1f} | "
                  f"V={features['vitesse_conv']:.2f}")
        except Exception as e:
            print(f"[!] Redis publish error: {e}")

    filter_bpf = f"tcp port {S7_PORT} or tcp port {MODBUS_PORT}"
    print(f"\n[PRODUCER] Listening on ports {S7_PORT} (S7comm) & {MODBUS_PORT} (Modbus TCP)...")
    print("[PRODUCER] Filter:", filter_bpf)
    print("[PRODUCER] Press Ctrl+C to stop\n")
    
    sniff(filter=filter_bpf, prn=process_packet, store=False)

# ==========================================
# 12. ALERT HISTORY MANAGER
# ==========================================
class AlertHistoryManager:
    """Stocke les alertes pour le résumé SOC."""
    
    def __init__(self, max_alerts=500, retention_hours=24):
        self.max_alerts = max_alerts
        self.retention_hours = retention_hours
        self.alerts = deque(maxlen=max_alerts)
        self.lock = threading.Lock()
        self.total_alerts = 0
        
    def add_alert(self, alert_data):
        """Ajoute une alerte avec timestamp."""
        with self.lock:
            if 'timestamp' not in alert_data:
                alert_data['timestamp'] = datetime.now().isoformat()
            self.alerts.append(alert_data)
            self.total_alerts += 1
            self._cleanup()
    
    def _cleanup(self):
        """Supprime les alertes > retention_hours."""
        cutoff = (datetime.now() - timedelta(hours=self.retention_hours)).isoformat()
        self.alerts = deque(
            [a for a in self.alerts if a.get('timestamp', '') > cutoff],
            maxlen=self.max_alerts
        )
    
    def get_recent(self, hours=24, limit=50):
        """Récupère les dernières alertes."""
        cutoff = (datetime.now() - timedelta(hours=hours)).isoformat()
        return [a for a in self.alerts if a.get('timestamp', '') > cutoff][-limit:]
    
    def get_stats(self):
        """Statistiques rapides."""
        recent = self.get_recent(24)
        return {
            'total': self.total_alerts,
            'active': len(self.alerts),
            'last_24h': len(recent),
            'first': self.alerts[0] if self.alerts else None,
            'last': self.alerts[-1] if self.alerts else None
        }
    
    def clear(self):
        with self.lock:
            self.alerts.clear()
            self.total_alerts = 0

# ==========================================
# 13. TELEGRAM COMMAND HANDLER
# ==========================================
class TelegramCommandHandler:
    """Gère les commandes Telegram: /summary, /clear, /help"""
    
    def __init__(self, history_manager, bot_token, chat_id):
        self.history = history_manager
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.ollama_url = "http://localhost:11434/api/generate"
        self.ollama_model = "llama3.2"
    
    def handle(self, command):
        """Traite une commande Telegram."""
        cmd = command.strip().lower()
        
        if cmd == "/summary":
            return self._summary()
        elif cmd == "/clear":
            return self._clear()
        elif cmd == "/help":
            return self._help()
        else:
            return "❌ Commande inconnue. Try /help "
    
    def _summary(self):
        """Génère un rapport SOC des 24h."""
        recent = self.history.get_recent(24, 30)
        
        if not recent:
            return "✅ Aucune alerte détectée dans les dernières 24 heures."
        
        # Statistiques
        total = len(recent)
        ips = {}
        types = {}
        
        for a in recent:
            ip = a.get('src_ip', '?')
            t = a.get('anomaly_type', 'Inconnu')
            ips[ip] = ips.get(ip, 0) + 1
            types[t] = types.get(t, 0) + 1
        
        top_ips = sorted(ips.items(), key=lambda x: x[1], reverse=True)[:3]
        top_types = sorted(types.items(), key=lambda x: x[1], reverse=True)[:3]
        
        # Prompt Ollama
        alerts_text = "\n".join([
            f"- {a.get('timestamp', '')[:16]} | {a.get('src_ip')} | {a.get('anomaly_type')}"
            for a in recent[-10:]
        ])
        
        prompt = f"""
Rapport SOC 24h:
- Alertes: {total}
- Top IPs: {', '.join([f'{ip}({c})' for ip,c in top_ips])}
- Top types: {', '.join([f'{t}({c})' for t,c in top_types])}

Alertes récentes:
{alerts_text}

Réponds en 3 lignes:
1. Tendance (stable/↑/↓)
2. Risque principal (une phrase)
3. Action recommandée (une phrase)
"""
        
        try:
            resp = requests.post(
                self.ollama_url,
                json={"model": self.ollama_model, "prompt": prompt, "stream": False, "temperature": 0.3},
                timeout=15
            )
            analysis = resp.json().get("response", "Analyse indisponible")
        except:
            analysis = "⚠️ Analyse Ollama indisponible."
        
        # Rapport formaté
        return f"""
📊 RÉSUMÉ SOC (24h)
═══════════════════
Alertes: {total}
Top IPs: {', '.join([f'{ip}({c})' for ip,c in top_ips])}
Top types: {', '.join([f'{t}({c})' for t,c in top_types])}

🔍 {analysis}
"""
    
    def _clear(self):
        self.history.clear()
        return "✅ Historique effacé."
    
    def _help(self):
        return """
📋 COMMANDES:
/summary - Rapport SOC 24h
/clear   - Effacer historique
/help    - Cette aide
"""
    
    def send_response(self, text):
        """Envoie la réponse sur Telegram."""
        url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
        try:
            requests.post(url, json={
                "chat_id": self.chat_id,
                "text": text,
                "parse_mode": "Markdown"
            }, timeout=5)
        except Exception as e:
            print(f"[!] Erreur Telegram: {e}")


# ===============================================
# 14. PROCESS 2: CONSUMER (REDIS TO ML INFERENCE)
# ===============================================

# Variables globales
alert_history = AlertHistoryManager(max_alerts=500, retention_hours=24)
command_handler = None

def run_detector_consumer():
    """Consumes messages from Redis, runs ML inference, and alerts."""
    global model_loader, alert_history, command_handler

    try:
        redis_ch = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=0)
        redis_ch.ping()
        print(f"[CONSUMER] Connected to Redis on {REDIS_HOST}:{REDIS_PORT}")
    except redis.ConnectionError:
        print(f"[!] Cannot connect to Redis/Memurai server")
        return

    pubsub = redis_ch.pubsub()
    pubsub.subscribe(REDIS_CHANNEL)

    # Initialiser le handler de commandes
    command_handler = TelegramCommandHandler(
        alert_history, 
        TELEGRAM_BOT_TOKEN, 
        TELEGRAM_CHAT_ID
    )

    # Load models
    model_loader = ModelLoader()
    if not model_loader.load_models(MODEL_TYPE):
        print("[CONSUMER] Running in heuristic mode (no ML model loaded).")
    
    print(f"[CONSUMER] Listening for telemetry on Redis channel '{REDIS_CHANNEL}'...")
    print(f"[CONSUMER] Model type: {MODEL_TYPE}")
    if MODEL_TYPE == "lstm_autoencoder":
        print(f"[CONSUMER] LSTM Buffer Size: {BUFFER_SIZE}")
        print(f"[CONSUMER] LSTM Threshold: {LSTM_THRESHOLD}")
    print()

    # Initialize packet buffer for LSTM
    packet_buffer = PacketBuffer(BUFFER_SIZE)
    anomaly_count = 0

    # Démarrer l'écoute des commandes Telegram dans un thread séparé
    def listen_telegram_commands():
        """Écoute les messages Telegram entrants."""
        offset = 0
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/getUpdates"
        
        while True:
            try:
                resp = requests.get(url, params={"offset": offset, "timeout": 30})
                if resp.status_code == 200:
                    for update in resp.json().get("result", []):
                        offset = update["update_id"] + 1
                        msg = update.get("message", {})
                        text = msg.get("text", "")
                        
                        if text.startswith("/"):
                            response = command_handler.handle(text)
                            command_handler.send_response(response)
                time.sleep(1)
            except Exception as e:
                print(f"[!] Telegram listener error: {e}")
                time.sleep(5)
    
    # Thread pour les commandes Telegram
    tg_thread = threading.Thread(target=listen_telegram_commands, daemon=True)
    tg_thread.start()
    print("[CONSUMER] Écoute commandes Telegram activée")

    # Traitement des alertes
    for message in pubsub.listen():
        if message["type"] == "message":
            try:
                alert_payload = json.loads(message["data"].decode("utf-8"))
            except Exception as e:
                print(f"[!] JSON decode error: {e}")
                continue

            # Add packet to buffer
            is_full = packet_buffer.add_packet(alert_payload)
            
            # For LSTM, we need a full buffer before predicting
            if MODEL_TYPE == "lstm_autoencoder" and not is_full:
                continue
            
            # Get the current feature vector or sequence
            is_anomaly = False
            prediction_label = 0
            anomaly_type = "Normal"
            recon_error = 0.0
            
            if MODEL_TYPE == "lstm_autoencoder" and is_full:
                sequence = packet_buffer.get_sequence()
                if sequence is not None:
                    prediction_label, is_anomaly, recon_error = model_loader.predict_lstm(sequence)
                    if is_anomaly:
                        anomaly_type = f"LSTM Anomaly (Recon Error: {recon_error:.4f})"
            elif MODEL_TYPE in ["isolation_forest", "xgboost"]:
                feature_vector = build_feature_vector(alert_payload)
                prediction, is_anomaly = model_loader.predict(feature_vector)
                if prediction is not None:
                    prediction_label = int(prediction) if prediction != -1 else 1
                    if is_anomaly:
                        anomaly_type = f"ML Anomaly (Label: {prediction_label})"
            elif MODEL_TYPE == "ensemble":
                feature_vector = build_feature_vector(alert_payload)
                prediction, is_anomaly = model_loader.predict(feature_vector)
                if prediction is not None:
                    prediction_label = int(prediction)
                    if is_anomaly:
                        anomaly_type = f"Ensemble Anomaly (Label: {prediction_label})"
            
            # Fallback to heuristic if no ML anomaly detected
            if not is_anomaly:
                is_heuristic, heuristic_type = check_heuristic_anomaly(alert_payload)
                if is_heuristic:
                    is_anomaly = True
                    anomaly_type = f"Heuristic: {heuristic_type}"

            if is_anomaly:
                anomaly_count += 1
                
                # Ajouter à l'historique
                alert_data = {
                    'timestamp': datetime.now().isoformat(),
                    'src_ip': alert_payload.get('src_ip'),
                    'dst_ip': alert_payload.get('dst_ip'),
                    'protocol': alert_payload.get('protocol'),
                    'pression': alert_payload.get('pression', 0),
                    'temp_four': alert_payload.get('temp_four', 0),
                    'vitesse_conv': alert_payload.get('vitesse_conv', 0),
                    'anomaly_type': anomaly_type,
                    'prediction_label': prediction_label,
                    'function_code': alert_payload.get('function_code', 0)
                }
                alert_history.add_alert(alert_data)
                
                print(f"\n[🚨 ANOMALY #{anomaly_count} - {alert_payload.get('protocol', 'Unknown')}]")
                print(f"    Type: {anomaly_type}")
                print(f"    Source: {alert_payload.get('src_ip', 'Unknown')} -> {alert_payload.get('dst_ip', 'Unknown')}")
                print(f"    Pression: {alert_payload.get('pression', 0):.2f} bar")
                print(f"    Température: {alert_payload.get('temp_four', 0):.1f} °C")
                print(f"    Vitesse: {alert_payload.get('vitesse_conv', 0):.2f} m/min")
                print(f"    Function Code: {hex(alert_payload.get('function_code', 0))}")
                if recon_error > 0:
                    print(f"    Reconstruction Error: {recon_error:.4f}")

                # ==========================================
                # ENVOI IMMÉDIAT + OLLAMA EN ARRIÈRE-PLAN
                # ==========================================
                
                immediate_summary = f"🚨 {anomaly_type} - Vérifier {alert_payload.get('src_ip', '?')}"
                send_telegram_alert(immediate_summary, alert_payload, prediction_label)
                print("[+] Alerte envoyée immédiatement à Telegram")

                # 2. ANALYSE OLLAMA EN ARRIÈRE-PLAN (non-bloquant)
                def process_ollama_background(payload, a_type, p_label):

                    print("[*] Analyse Ollama en arrière-plan démarrée...")
                    try:
                        summary = query_ollama_summary(payload, a_type, p_label)
                        
                        # Envoyer le suivi avec l'analyse complète
                        followup_message = (
                            f"🧠 <b>Analyse Ollama</b>\n\n"
                            f"{summary}"
                        )
                        send_telegram_alert(followup_message, payload, p_label)
                        print("[+] Analyse Ollama envoyée en suivi")
                    except Exception as e:
                        print(f"[!] Erreur analyse Ollama: {e}")
                
                # Démarrer le thread en arrière-plan
                threading.Thread(
                    target=process_ollama_background,
                    args=(alert_payload, anomaly_type, prediction_label),
                    daemon=True
                ).start()
                print("[*] Traitement Ollama en arrière-plan (non-bloquant)")
                
                # Clear buffer after anomaly detection for LSTM
                if MODEL_TYPE == "lstm_autoencoder":
                    packet_buffer.clear()

    print(f"\n[CONSUMER] Stopped. Total anomalies detected: {anomaly_count}")


# ==========================================
# 15. TEST MODE (PCAP REPLAY)
# ==========================================
    """
    [+] Test mode instead of live sniffing
    [+] Validate Model Performance
    [+] Continuous Integration / Testing
    [+] Debugging False Positives
    [+] Capture suspicious traffic to PCAP
    """

def run_test_mode(pcap_file=None):
    
    from scapy.all import rdpcap
    
    if pcap_file is None:
        pcap_file = "../Datasets_wireshark/modbus_cap.pcapng"
    
    print(f"\n[TEST MODE] Processing PCAP: {pcap_file}")
    
    try:
        packets = rdpcap(pcap_file)
        print(f"[TEST] Loaded {len(packets)} packets")
    except Exception as e:
        print(f"[!] Error loading PCAP: {e}")
        return
    
    global model_loader
    model_loader = ModelLoader()
    model_loader.load_models(MODEL_TYPE)
    
    packet_buffer = PacketBuffer(BUFFER_SIZE)
    telemetry_count = 0
    anomaly_count = 0
    
    for idx, packet in enumerate(packets):
        if idx % 100 == 0:
            print(f"[TEST] Processed {idx}/{len(packets)} packets")
        
        if not packet.haslayer(TCP) or not packet.haslayer(Raw):
            continue
        
        dport = packet[TCP].dport
        sport = packet[TCP].sport
        features = None
        
        if dport == S7_PORT or sport == S7_PORT:
            features = extract_s7_features(packet)
        elif dport == MODBUS_PORT or sport == MODBUS_PORT:
            features = extract_modbus_features(packet)
        
        if not features or not features.get("has_telemetry", False):
            continue
        
        telemetry_count += 1
        
        is_full = packet_buffer.add_packet(features)
        
        if MODEL_TYPE == "lstm_autoencoder" and not is_full:
            continue
        
        is_anomaly = False
        anomaly_type = "Normal"
        
        if MODEL_TYPE == "lstm_autoencoder" and is_full:
            sequence = packet_buffer.get_sequence()
            if sequence is not None:
                prediction, is_anomaly, recon_error = model_loader.predict_lstm(sequence)
                if is_anomaly:
                    anomaly_type = f"LSTM (Error: {recon_error:.4f})"
        else:
            feature_vector = build_feature_vector(features)
            prediction, is_anomaly = model_loader.predict(feature_vector)
            if is_anomaly:
                anomaly_type = "ML Anomaly"
        
        if not is_anomaly:
            is_heuristic, heuristic_type = check_heuristic_anomaly(features)
            if is_heuristic:
                is_anomaly = True
                anomaly_type = f"Heuristic: {heuristic_type}"
        
        if is_anomaly:
            anomaly_count += 1
            print(f"\n[🚨 TEST ANOMALY #{anomaly_count}]")
            print(f"    Type: {anomaly_type}")
            print(f"    Protocol: {features.get('protocol', 'Unknown')}")
            print(f"    Pression: {features.get('pression', 0):.2f} bar")
            print(f"    Température: {features.get('temp_four', 0):.1f} °C")
            print(f"    Vitesse: {features.get('vitesse_conv', 0):.2f} m/min")
    
    print(f"\n[TEST] Complete:")
    print(f"    Total packets: {len(packets)}")
    print(f"    Telemetry packets: {telemetry_count}")
    print(f"    Anomalies detected: {anomaly_count}")

# ==========================================
# 16. ENTRY POINT
# ==========================================
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="ICS Anomaly Detector")
    parser.add_argument("--test", action="store_true", help="Run in test mode with PCAP replay")
    parser.add_argument("--pcap", type=str, help="PCAP file for test mode")
    parser.add_argument("--model", type=str, default="lstm_autoencoder", 
                       choices=["isolation_forest", "xgboost", "lstm_autoencoder", "ensemble"],
                       help="Model type to use")
    parser.add_argument("--threshold", type=float, default=1.5,
                       help="Anomaly threshold for LSTM (higher = less sensitive)")
    args = parser.parse_args()
    
    MODEL_TYPE = args.model
    if args.threshold:
        LSTM_THRESHOLD = args.threshold
    
    if args.test:
        run_test_mode(args.pcap)
    else:
        print("\n" + "="*70)
        print("ICS/SCADA Anomaly Detector")
        print("="*70)
        print(f"Model Type: {MODEL_TYPE}")
        print(f"Redis Channel: {REDIS_CHANNEL}")
        if MODEL_TYPE == "lstm_autoencoder":
            print(f"LSTM Buffer Size: {BUFFER_SIZE}")
            print(f"LSTM Threshold: {LSTM_THRESHOLD}")
        print("="*70 + "\n")
        
        producer_process = multiprocessing.Process(target=run_sniffer_producer)
        consumer_process = multiprocessing.Process(target=run_detector_consumer)

        try:
            producer_process.start()
            consumer_process.start()
            producer_process.join()
            consumer_process.join()
        except KeyboardInterrupt:
            print("\n[*] Shutting down sniffer and detector processes...")
            producer_process.terminate()
            consumer_process.terminate()
            producer_process.join()
            consumer_process.join()
            print("[*] Shutdown complete.")
