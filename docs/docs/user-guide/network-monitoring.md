# Network Monitoring

This guide covers real-time network interface monitoring using RL-IDS.

## Overview

The `network_monitor.py` script provides comprehensive real-time monitoring of network interfaces, analyzing all network traffic and detecting potential intrusions using the trained DQN model.

## Features

- **Real-time packet capture** using Scapy
- **CICIDS2017 feature extraction** from network flows
- **DQN-based threat detection** via API integration
- **Flow-based analysis** with stateful connection tracking
- **Interactive terminal interface** with live statistics
- **Configurable attack filtering** and thresholds
- **Comprehensive logging** with JSON and text formats

## Basic Usage

### Start Monitoring

```bash
# Monitor default interface (auto-detected)
sudo python network_monitor.py

# Monitor specific interface
sudo python network_monitor.py eth0

# Monitor with custom API endpoint
sudo python network_monitor.py wlan0 --api-url http://localhost:8000
```

### Command Line Options

```bash
python network_monitor.py [INTERFACE] [OPTIONS]

Arguments:
  INTERFACE        Network interface to monitor (default: auto-detect)

Options:
  --api-url TEXT   RL-IDS API endpoint URL [default: http://localhost:8000]
  --help          Show help message and exit
```

## User Interface

The monitor provides a real-time terminal interface with:

### Statistics Panel
- **Uptime**: How long the monitor has been running
- **Packets**: Total packets processed
- **Rate**: Packets per minute
- **Active Flows**: Number of tracked network flows
- **Attacks**: Total attacks detected
- **Ignored**: Attacks filtered out based on configuration
- **Queue Size**: Async processing queue size

### Configuration Panel
- **Interface**: Currently monitored network interface
- **API URL**: RL-IDS API endpoint
- **Threshold**: Confidence threshold for attack detection
- **Status**: Current monitoring status

### Recent Alerts
Real-time display of detected attacks with:
- Timestamp
- Attack type
- Source IP address
- Confidence score

## Network Flow Analysis

The monitor implements stateful flow tracking:

### Flow Identification
Flows are identified by the 5-tuple:
- Source IP
- Destination IP  
- Source Port
- Destination Port
- Protocol

### Flow Statistics
For each flow, the system tracks:
- **Packet counts** (forward/backward direction)
- **Byte counts** (forward/backward direction)
- **Timing information** (start time, inter-arrival times)
- **TCP flags** (if applicable)
- **Packet sizes** and statistical measures

### Feature Extraction
The `CICIDSFeatureExtractor` class extracts 78 features from each flow:

```python
class CICIDSFeatureExtractor:
    """Extract CICIDS2017-compatible features from network packets"""
    
    def __init__(self):
        self.feature_names = [
            'flow_duration', 'total_fwd_packets', 'total_bwd_packets',
            'total_length_fwd_packets', 'total_length_bwd_packets',
            'fwd_packet_length_max', 'fwd_packet_length_min', 
            'fwd_packet_length_mean', 'fwd_packet_length_std',
            # ... 69 more features
        ]
    
    def extract_features(self, packet, flow_data):
        """Extract 78 CICIDS2017 features from packet and flow data"""
        # Implementation extracts statistical features from flow
        # Returns list of 78 float values
```

## Detection Process

### 1. Packet Capture
- Captures packets on specified interface using Scapy
- Filters packets based on configurable criteria
- Handles multiple protocols (TCP, UDP, ICMP)

### 2. Flow Processing
- Groups packets into bidirectional flows
- Maintains flow state and statistics
- Calculates timing and size metrics

### 3. Feature Extraction
- Extracts 78 CICIDS2017 features per flow
- Handles edge cases and missing data
- Normalizes feature values

### 4. Threat Detection
- Sends features to RL-IDS API for classification
- Receives prediction with confidence score
- Applies configured threshold for alerting

### 5. Alert Processing
- Logs detected attacks with full context
- Applies ignore filters for known false positives
- Updates real-time interface with new alerts

## Configuration

### Attack Filtering
Configure which attack types to ignore:

```python
self.ignored_attacks = [
    'heartbleed',    # Often false positives
    'portscan',      # Too noisy for some environments
    'benign'         # Normal traffic
]
```

### Confidence Threshold
Set minimum confidence for attack alerts:

```python
self.confidence_threshold = 0.7  # 70% confidence minimum
```

### Network Interface Selection
The monitor can auto-detect the primary network interface or use a specified one:

```python
def get_available_interfaces(self):
    """Get list of available network interfaces with IP addresses"""
    interfaces = []
    for iface, addrs in psutil.net_if_addrs().items():
        for addr in addrs:
            if addr.family.name == 'AF_INET' and not addr.address.startswith('127.'):
                interfaces.append(iface)
    return list(set(interfaces))
```

## Logging

The monitor generates several log files:

### `network_monitor.log`
General operational logs:
```
2025-06-27 10:30:15 | INFO | Starting network monitoring on interface wlan0
2025-06-27 10:30:16 | INFO | API connection established: http://localhost:8000
2025-06-27 10:30:45 | WARNING | High packet rate detected: 150 pps
```

### `intrusion_alerts.log`
Detailed attack information:
```
2025-06-27 10:31:22 | CRITICAL | ATTACK DETECTED: DoS Hulk
  Source: 192.168.1.100:45123 -> 192.168.1.1:80
  Confidence: 87.5%
  Flow Duration: 12.3s
  Features: [0.123, 0.456, ...]
```

### `alerts.json`
Machine-readable alert data:
```json
{
  "timestamp": "2025-06-27T10:31:22.123456",
  "attack_type": "DoS Hulk",
  "source_ip": "192.168.1.100",
  "dest_ip": "192.168.1.1",
  "confidence": 0.875,
  "flow_key": "192.168.1.100:45123-192.168.1.1:80-TCP",
  "features": [0.123, 0.456, ...]
}
```

## Troubleshooting

### Common Issues

**Permission Denied**
```bash
# Solution: Run with sudo or set capabilities
sudo python network_monitor.py
# OR
sudo setcap cap_net_raw,cap_net_admin=eip $(which python3)
```

**Interface Not Found**
```bash
# List available interfaces
python -c "import psutil; print(list(psutil.net_if_addrs().keys()))"
```

**API Connection Failed**
```bash
# Check API server status
curl http://localhost:8000/health
```

**High Memory Usage**
- Reduce `max_flows` parameter
- Increase `flow_timeout` for faster cleanup
- Add more attack types to ignore list

**Packet Drops**
- Increase system buffer sizes
- Reduce feature extraction frequency
- Use dedicated monitoring hardware

### Debug Mode
Enable verbose logging for troubleshooting:

```bash
# Set environment variable for debug logging
export RLIDS_DEBUG=true
export RLIDS_LOG_LEVEL=DEBUG

sudo python network_monitor.py wlan0
```