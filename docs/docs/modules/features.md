# Feature Extraction

This document covers the CICIDS2017 feature extraction system used in RL-IDS for converting network packets into standardized feature vectors.

## Overview

The feature extraction system (`network_monitor.py` - `CICIDSFeatureExtractor` class) converts raw network packets into 78 standardized features compatible with the CICIDS2017 dataset. This enables the DQN models to analyze real-time network traffic using the same feature space they were trained on.

## CICIDS2017 Feature Set

The system extracts 78 features organized into several categories:

### Flow Duration Features
- `flow_duration` - Duration of the network flow in seconds

### Packet Count Features
- `total_fwd_packets` - Total packets in forward direction
- `total_bwd_packets` - Total packets in backward direction

### Byte Count Features
- `total_length_fwd_packets` - Total bytes in forward direction
- `total_length_bwd_packets` - Total bytes in backward direction

### Packet Length Statistics
Forward direction:
- `fwd_packet_length_max` - Maximum packet length
- `fwd_packet_length_min` - Minimum packet length
- `fwd_packet_length_mean` - Mean packet length
- `fwd_packet_length_std` - Standard deviation of packet lengths

Backward direction:
- `bwd_packet_length_max` - Maximum packet length
- `bwd_packet_length_min` - Minimum packet length
- `bwd_packet_length_mean` - Mean packet length
- `bwd_packet_length_std` - Standard deviation of packet lengths

### Flow Rate Features
- `flow_bytes_per_sec` - Bytes per second in the flow
- `flow_packets_per_sec` - Packets per second in the flow
- `fwd_packets_per_sec` - Forward packets per second
- `bwd_packets_per_sec` - Backward packets per second

### Inter-Arrival Time Features
- `flow_iat_mean` - Mean inter-arrival time
- `flow_iat_std` - Standard deviation of inter-arrival times
- `flow_iat_max` - Maximum inter-arrival time
- `flow_iat_min` - Minimum inter-arrival time

### TCP Flag Features
- `fin_flag_count` - Count of FIN flags
- `syn_flag_count` - Count of SYN flags
- `rst_flag_count` - Count of RST flags
- `psh_flag_count` - Count of PSH flags
- `ack_flag_count` - Count of ACK flags
- `urg_flag_count` - Count of URG flags

### Additional Statistical Features
- `min_packet_length` - Minimum packet length across all packets
- `max_packet_length` - Maximum packet length across all packets
- `packet_length_mean` - Mean packet length
- `packet_length_std` - Standard deviation of packet lengths
- `packet_length_variance` - Variance of packet lengths

## CICIDSFeatureExtractor Implementation

### Class Structure

```python
class CICIDSFeatureExtractor:
    """Extract CICIDS2017-compatible features from network packets"""
    
    def __init__(self):
        self.feature_names = [
            'flow_duration', 'total_fwd_packets', 'total_bwd_packets',
            'total_length_fwd_packets', 'total_length_bwd_packets',
            'fwd_packet_length_max', 'fwd_packet_length_min', 'fwd_packet_length_mean',
            'fwd_packet_length_std', 'bwd_packet_length_max', 'bwd_packet_length_min',
            'bwd_packet_length_mean', 'bwd_packet_length_std', 'flow_bytes_per_sec',
            'flow_packets_per_sec', 'flow_iat_mean', 'flow_iat_std', 'flow_iat_max',
            'flow_iat_min', 'fwd_iat_total', 'fwd_iat_mean', 'fwd_iat_std',
            'fwd_iat_max', 'fwd_iat_min', 'bwd_iat_total', 'bwd_iat_mean',
            'bwd_iat_std', 'bwd_iat_max', 'bwd_iat_min', 'fwd_psh_flags',
            'bwd_psh_flags', 'fwd_urg_flags', 'bwd_urg_flags', 'fwd_header_length',
            'bwd_header_length', 'fwd_packets_per_sec', 'bwd_packets_per_sec',
            'min_packet_length', 'max_packet_length', 'packet_length_mean',
            'packet_length_std', 'packet_length_variance', 'fin_flag_count',
            'syn_flag_count', 'rst_flag_count', 'psh_flag_count', 'ack_flag_count',
            'urg_flag_count', 'cwe_flag_count', 'ece_flag_count', 'down_up_ratio',
            'average_packet_size', 'avg_fwd_segment_size', 'avg_bwd_segment_size',
            'fwd_header_length_2', 'fwd_avg_bytes_per_bulk', 'fwd_avg_packets_per_bulk',
            'fwd_avg_bulk_rate', 'bwd_avg_bytes_per_bulk', 'bwd_avg_packets_per_bulk',
            'bwd_avg_bulk_rate', 'subflow_fwd_packets', 'subflow_fwd_bytes',
            'subflow_bwd_packets', 'subflow_bwd_bytes', 'init_win_bytes_forward',
            'init_win_bytes_backward', 'act_data_pkt_fwd', 'min_seg_size_forward',
            'active_mean', 'active_std', 'active_max', 'active_min', 'idle_mean',
            'idle_std', 'idle_max', 'idle_min'
        ]
```

### Feature Extraction Process

```python
def extract_features(self, packet, flow_data):
    """Extract 78 CICIDS2017 features from packet and flow data"""
    try:
        features = [0.0] * 78
        
        if IP not in packet:
            return features
        
        current_time = datetime.now()
        packets = flow_data.get('packets', [])
        
        if not packets:
            return features
        
        # Basic flow statistics
        flow_start = flow_data.get('start_time', current_time)
        flow_duration = (current_time - flow_start).total_seconds()
        
        # Packet counts
        fwd_packets = flow_data.get('forward_packets', 0)
        bwd_packets = flow_data.get('backward_packets', 0)
        total_packets = fwd_packets + bwd_packets
        
        # Byte counts
        fwd_bytes = flow_data.get('forward_bytes', 0)
        bwd_bytes = flow_data.get('backward_bytes', 0)
        total_bytes = fwd_bytes + bwd_bytes
        
        # Extract packet sizes by direction
        fwd_sizes = [p['size'] for p in packets if p.get('direction') == 'forward']
        bwd_sizes = [p['size'] for p in packets if p.get('direction') == 'backward']
        all_sizes = [p['size'] for p in packets]
        
        # Feature extraction
        features[0] = flow_duration
        features[1] = float(fwd_packets)
        features[2] = float(bwd_packets)
        features[3] = float(fwd_bytes)
        features[4] = float(bwd_bytes)
        
        # Forward packet statistics
        if fwd_sizes:
            features[5] = float(max(fwd_sizes))
            features[6] = float(min(fwd_sizes))
            features[7] = float(np.mean(fwd_sizes))
            features[8] = float(np.std(fwd_sizes))
        
        # Backward packet statistics
        if bwd_sizes:
            features[9] = float(max(bwd_sizes))
            features[10] = float(min(bwd_sizes))
            features[11] = float(np.mean(bwd_sizes))
            features[12] = float(np.std(bwd_sizes))
        
        # Flow rates
        if flow_duration > 0:
            features[13] = float(total_bytes / flow_duration)
            features[14] = float(total_packets / flow_duration)
            features[35] = float(fwd_packets / flow_duration)
            features[36] = float(bwd_packets / flow_duration)
        
        # Inter-arrival times
        if len(packets) > 1:
            iats = []
            for i in range(1, len(packets)):
                iat = (packets[i]['timestamp'] - packets[i-1]['timestamp']).total_seconds()
                iats.append(iat)
            
            if iats:
                features[15] = float(np.mean(iats))
                features[16] = float(np.std(iats))
                features[17] = float(max(iats))
                features[18] = float(min(iats))
        
        # Packet length statistics
        if all_sizes:
            features[37] = float(min(all_sizes))
            features[38] = float(max(all_sizes))
            features[39] = float(np.mean(all_sizes))
            features[40] = float(np.std(all_sizes))
            features[41] = float(np.var(all_sizes))
            features[49] = float(np.mean(all_sizes))
        
        # TCP flags analysis
        if TCP in packet:
            tcp_flags = flow_data.get('tcp_flags', [])
            if tcp_flags:
                features[42] = float(sum(1 for f in tcp_flags if int(f) & 0x01))  # fin
                features[43] = float(sum(1 for f in tcp_flags if int(f) & 0x02))  # syn
                features[44] = float(sum(1 for f in tcp_flags if int(f) & 0x04))  # rst
                features[45] = float(sum(1 for f in tcp_flags if int(f) & 0x08))  # psh
                features[46] = float(sum(1 for f in tcp_flags if int(f) & 0x10))  # ack
                features[47] = float(sum(1 for f in tcp_flags if int(f) & 0x20))  # urg
        
        # Additional features
        features[50] = float(np.mean(fwd_sizes) if fwd_sizes else 0)
        features[51] = float(np.mean(bwd_sizes) if bwd_sizes else 0)
        
        # Normalize and clean values
        features = [min(max(f, -1e6), 1e6) for f in features]
        features = [0.0 if np.isnan(f) or np.isinf(f) else f for f in features]
        
        return features
        
    except Exception as e:
        return [0.0] * 78
```

## Flow Tracking

### Flow Identification

Network flows are identified using a 5-tuple:

```python
def get_flow_key(self, packet):
    """Generate unique flow key from packet"""
    if IP not in packet:
        return None
    
    src_ip = packet[IP].src
    dst_ip = packet[IP].dst
    protocol = packet[IP].proto
    
    src_port = 0
    dst_port = 0
    
    if TCP in packet:
        src_port = packet[TCP].sport
        dst_port = packet[TCP].dport
        protocol_name = "TCP"
    elif UDP in packet:
        src_port = packet[UDP].sport
        dst_port = packet[UDP].dport
        protocol_name = "UDP"
    else:
        protocol_name = "OTHER"
    
    # Create bidirectional flow key (normalize direction)
    if (src_ip, src_port) < (dst_ip, dst_port):
        flow_key = f"{src_ip}:{src_port}-{dst_ip}:{dst_port}-{protocol_name}"
    else:
        flow_key = f"{dst_ip}:{dst_port}-{src_ip}:{src_port}-{protocol_name}"
    
    return flow_key
```

### Flow State Management

```python
class FlowTracker:
    def __init__(self):
        self.flows = {}
        self.flow_timeout = 120  # seconds
    
    def update_flow(self, packet, flow_key):
        """Update flow statistics with new packet"""
        current_time = datetime.now()
        
        if flow_key not in self.flows:
            self.flows[flow_key] = {
                'start_time': current_time,
                'last_seen': current_time,
                'packets': [],
                'forward_packets': 0,
                'backward_packets': 0,
                'forward_bytes': 0,
                'backward_bytes': 0,
                'tcp_flags': []
            }
        
        flow = self.flows[flow_key]
        flow['last_seen'] = current_time
        
        # Determine packet direction
        direction = self.get_packet_direction(packet, flow_key)
        
        # Update packet statistics
        packet_size = len(packet)
        packet_info = {
            'timestamp': current_time,
            'size': packet_size,
            'direction': direction
        }
        flow['packets'].append(packet_info)
        
        # Update counters
        if direction == 'forward':
            flow['forward_packets'] += 1
            flow['forward_bytes'] += packet_size
        else:
            flow['backward_packets'] += 1
            flow['backward_bytes'] += packet_size
        
        # Extract TCP flags if present
        if TCP in packet:
            flow['tcp_flags'].append(packet[TCP].flags)
        
        return flow
    
    def cleanup_expired_flows(self):
        """Remove expired flows to prevent memory leaks"""
        current_time = datetime.now()
        expired_flows = []
        
        for flow_key, flow in self.flows.items():
            if (current_time - flow['last_seen']).total_seconds() > self.flow_timeout:
                expired_flows.append(flow_key)
        
        for flow_key in expired_flows:
            del self.flows[flow_key]
        
        return len(expired_flows)
```

## Data Preprocessing

### CICIDS2017 Dataset Processing

The `rl_ids/make_dataset.py` module handles preprocessing of the original CICIDS2017 CSV files:

```python
class DataGenerator:
    """Handles loading and initial preprocessing of raw CSV data files"""

    def __init__(self):
        self.label_encoder = None
        self.processed_data = None

    def load_and_preprocess_data(self, data_dir: Path = RAW_DATA_DIR) -> pd.DataFrame:
        """Load and preprocess CSV data files from the specified directory."""
        
        # Find CSV files
        csv_files = [f for f in os.listdir(data_dir) if f.endswith(".csv")]
        
        # Load CSV files with progress tracking
        data_frames = []
        for csv_file in tqdm(csv_files, desc="Loading CSV files"):
            file_path = data_dir / csv_file
            try:
                df = pd.read_csv(file_path)
                data_frames.append(df)
            except Exception as e:
                logger.error(f"Failed to load {csv_file}: {e}")
        
        # Combine all dataframes
        combined_df = pd.concat(data_frames, ignore_index=True)
        
        return combined_df
```

### Feature Normalization

```python
def normalize_features(self, df: pd.DataFrame, scaler_type: str = "standard") -> pd.DataFrame:
    """Normalize feature columns using specified scaler"""
    
    feature_columns = [col for col in df.columns if col not in ["Label", "Label_Original"]]
    
    if scaler_type == "standard":
        scaler = StandardScaler()
    elif scaler_type == "minmax":
        scaler = MinMaxScaler()
    elif scaler_type == "robust":
        scaler = RobustScaler()
    else:
        raise ValueError(f"Unknown scaler type: {scaler_type}")
    
    # Fit and transform features
    df_normalized = df.copy()
    df_normalized[feature_columns] = scaler.fit_transform(df[feature_columns])
    
    return df_normalized, scaler
```

### Label Encoding

```python
def encode_labels(self, df: pd.DataFrame) -> pd.DataFrame:
    """Encode string labels to numeric values"""
    
    # Store original labels
    df['Label_Original'] = df['Label'].copy()
    
    # Encode labels to numeric values
    self.label_encoder = LabelEncoder()
    df['Label'] = self.label_encoder.fit_transform(df['Label'])
    
    # Log label mapping
    label_mapping = dict(zip(self.label_encoder.classes_, 
                           self.label_encoder.transform(self.label_encoder.classes_)))
    logger.info(f"Label mapping: {label_mapping}")
    
    return df
```

## Real-time Feature Pipeline

### Integration with Network Monitor

The feature extraction integrates seamlessly with the network monitoring system:

```python
class RealTimeNetworkMonitor:
    def __init__(self, interface="wlan0", api_url="http://localhost:8000"):
        self.interface = interface
        self.api_url = api_url
        self.feature_extractor = CICIDSFeatureExtractor()
        self.flow_tracker = FlowTracker()
        
    def process_packet(self, packet):
        """Process captured packet for threat detection"""
        try:
            # Get flow key
            flow_key = self.get_flow_key(packet)
            if not flow_key:
                return
            
            # Update flow tracking
            flow_data = self.flow_tracker.update_flow(packet, flow_key)
            
            # Extract features
            features = self.feature_extractor.extract_features(packet, flow_data)
            
            # Send to API for analysis
            asyncio.create_task(self.analyze_features(features, flow_key))
            
        except Exception as e:
            logger.error(f"Error processing packet: {e}")
```

### Performance Optimizations

```python
# Efficient feature caching
class FeatureCache:
    def __init__(self, max_size=1000):
        self.cache = {}
        self.max_size = max_size
        self.access_times = {}
    
    def get_features(self, flow_key, flow_data):
        """Get cached features or compute new ones"""
        cache_key = self.compute_cache_key(flow_data)
        
        if cache_key in self.cache:
            self.access_times[cache_key] = time.time()
            return self.cache[cache_key]
        
        # Compute new features
        features = self.feature_extractor.extract_features(None, flow_data)
        
        # Cache with LRU eviction
        if len(self.cache) >= self.max_size:
            self.evict_lru()
        
        self.cache[cache_key] = features
        self.access_times[cache_key] = time.time()
        
        return features
```

## Feature Quality and Validation

### Feature Validation

```python
def validate_features(self, features):
    """Validate extracted features for quality and consistency"""
    
    # Check feature count
    if len(features) != 78:
        raise ValueError(f"Expected 78 features, got {len(features)}")
    
    # Check for invalid values
    for i, feature in enumerate(features):
        if np.isnan(feature) or np.isinf(feature):
            logger.warning(f"Invalid value in feature {i}: {feature}")
            features[i] = 0.0
        
        # Check reasonable ranges
        if abs(feature) > 1e6:
            logger.warning(f"Extreme value in feature {i}: {feature}")
            features[i] = np.clip(feature, -1e6, 1e6)
    
    return features
```

### Feature Statistics

```python
def compute_feature_statistics(self, feature_batches):
    """Compute statistics for extracted features"""
    
    features_array = np.array(feature_batches)
    
    stats = {
        'mean': np.mean(features_array, axis=0),
        'std': np.std(features_array, axis=0),
        'min': np.min(features_array, axis=0),
        'max': np.max(features_array, axis=0),
        'zero_ratio': np.mean(features_array == 0, axis=0)
    }
    
    return stats
```
