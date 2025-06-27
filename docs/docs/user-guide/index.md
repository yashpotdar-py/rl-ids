# User Guide

This guide covers the practical usage of RL-IDS for network monitoring and threat detection.

## Overview

RL-IDS provides three main operational modes:

1. **Network Interface Monitoring** - Monitor all traffic on a network interface
2. **Website-Specific Monitoring** - Monitor traffic to/from specific domains
3. **API Integration** - Use the REST API for custom integrations

## Network Interface Monitoring

Monitor all network traffic on a specific interface:

```bash
# Monitor default interface (auto-detected)
sudo python network_monitor.py

# Monitor specific interface
sudo python network_monitor.py eth0

# Monitor with custom API endpoint
sudo python network_monitor.py wlan0 --api-url http://localhost:8000
```

### Available Interfaces

To see available network interfaces:

```python
import psutil
for interface, addrs in psutil.net_if_addrs().items():
    print(f"Interface: {interface}")
    for addr in addrs:
        if addr.family.name == 'AF_INET':
            print(f"  IP: {addr.address}")
```

## Website-Specific Monitoring

Monitor traffic to/from specific websites:

```bash
# Monitor a specific domain
python website_monitor.py example.com

# Monitor with custom settings
python website_monitor.py example.com --api-url http://localhost:8000 --interface wlan0
```

This mode:
- Resolves domain names to IP addresses
- Generates test traffic to the target
- Captures and analyzes responses
- Detects potential attacks in the communication

## Configuration Options

### Environment Variables

Create a `.env` file based on .env.example:

```bash
cp .env.example .env
```

Key configuration options:
- API endpoint URLs
- Model file paths
- Logging levels
- Network interface preferences

.env.example:
```bash
# RL-IDS API Environment Configuration
# Copy this file to .env and modify as needed

# API Settings
RLIDS_APP_NAME=RL-IDS API
RLIDS_APP_VERSION=1.2.0
RLIDS_DEBUG=false

# Server Settings
RLIDS_HOST=0.0.0.0
RLIDS_PORT=8000
RLIDS_WORKERS=1

# Model Settings
RLIDS_MODEL_PATH=models/dqn_model_final.pt
RLIDS_DATA_PATH=data/processed/cicids2017_normalised.csv

# Performance Settings
RLIDS_MAX_BATCH_SIZE=100
RLIDS_PREDICTION_TIMEOUT=30.0

# Logging Settings
RLIDS_LOG_LEVEL=INFO
RLIDS_LOG_FORMAT={time} | {level} | {message}

# CORS Settings (for production, restrict these)
RLIDS_CORS_ORIGINS=["*"]
RLIDS_CORS_METHODS=["*"]
RLIDS_CORS_HEADERS=["*"]

# Rate Limiting
RLIDS_RATE_LIMIT_ENABLED=false
RLIDS_RATE_LIMIT_REQUESTS=100
RLIDS_RATE_LIMIT_WINDOW=60

# Health Check Settings
RLIDS_HEALTH_CHECK_TIMEOUT=5.0
```

### Runtime Parameters

Most scripts accept command-line arguments:

```bash
# Network monitor options
python network_monitor.py --help

# Website monitor options  
python website_monitor.py --help

# API server options
python run_api.py --help
```

## Understanding Output

### Real-time Monitor Display

The network monitor shows:

```
🛡️  RL-IDS NETWORK MONITOR 🛡️
================================================================================

📈 STATISTICS:
  📊 Uptime:        0:05:23
  📦 Packets:       1,247
  ⚡ Rate:          13.2/min
  🔍 Active Flows:  8
  🚨 Attacks:       2
  🔇 Ignored:       1
  📤 Queue Size:    0

🔧 CONFIGURATION:
  📡 Interface:     wlan0
  🌐 API URL:       http://localhost:8000
  ⚙️ Threshold:     70.0%
  🔧 Status:        Monitoring Active 🟢

🚨 RECENT ALERTS:
  | Time     | Attack Type | Source IP     | Confidence |
  | -------- | ----------- | ------------- | ---------- |
  | 14:23:15 | DoS Hulk    | 192.168.1.100 | 85%        |
  | 14:22:08 | Port Scan   | 10.0.0.15     | 92%        |
```

### Log Files

RL-IDS generates several log files in the `logs/` directory:

- `network_monitor.log` - General monitoring logs
- `intrusion_alerts.log` - Detected attack details
- `alerts.json` - Machine-readable alert data
- `website_monitor.log` - Website monitoring logs
- `ignored_attacks.json` - Filtered/ignored attacks

## Troubleshooting

### Permission Issues

Network monitoring requires elevated privileges:

```bash
# Run with sudo
sudo python network_monitor.py

# Or configure capabilities (Linux)
sudo setcap cap_net_raw,cap_net_admin=eip /usr/bin/python3
```

### Interface Not Found

List available interfaces:

```bash
python -c "
import psutil
interfaces = [iface for iface, addrs in psutil.net_if_addrs().items() 
              if any(addr.family.name == 'AF_INET' for addr in addrs)]
print('Available interfaces:', interfaces)
"
```

### API Connection Issues

Check if the API server is running:

```bash
# Test API health
curl http://localhost:8000/health

# Check API documentation
curl http://localhost:8000/docs
```

### High Memory Usage

For long-running monitoring:

1. Monitor the packet queue size in the UI
2. Adjust `confidence_threshold` to reduce false positives
3. Add more attack types to `ignored_attacks` list
4. Increase `cleanup_interval` for flow cleanup

## Advanced Usage

### Custom Feature Extraction

The system uses CICIDS2017-compatible features. To add custom features:

```python
from network_monitor import CICIDSFeatureExtractor

extractor = CICIDSFeatureExtractor()
features = extractor.extract_features(packet, flow_data)
```

### Filtering Traffic

Modify packet filters in `network_monitor.py`:

```python
# Example: Monitor only HTTP traffic
packet_filter = "tcp port 80 or tcp port 443"

# Example: Monitor specific subnet
packet_filter = "net 192.168.1.0/24"
```

### Custom Attack Types

Add or remove attack types from the ignore list:

```python
# In RealTimeNetworkMonitor.__init__()
self.ignored_attacks = ['heartbleed', 'portscan', 'benign']
```