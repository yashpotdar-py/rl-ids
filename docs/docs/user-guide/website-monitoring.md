# Website Monitoring

This guide covers website-specific traffic monitoring using RL-IDS to analyze communication with specific domains.

## Overview

The `website_monitor.py` script provides targeted monitoring of network traffic to and from specific websites. It generates controlled traffic, captures the responses, and analyzes the communication patterns for potential threats.

## Features

- **Domain-specific monitoring** with automatic IP resolution
- **Controlled traffic generation** using HTTP requests
- **Packet capture and analysis** for target communications
- **Real-time threat detection** via RL-IDS API integration
- **Interactive monitoring interface** with live statistics
- **Comprehensive logging** of traffic and detections

## Basic Usage

### Start Website Monitoring

```bash
# Monitor a specific domain
python website_monitor.py example.com

# Monitor with custom API endpoint
python website_monitor.py example.com --api-url http://localhost:8000

# Monitor with specific network interface
python website_monitor.py example.com --interface wlan0
```

### Command Line Options

```bash
python website_monitor.py TARGET [OPTIONS]

Arguments:
  TARGET           Target domain/website to monitor (required)

Options:
  --api-url TEXT   RL-IDS API endpoint URL [default: http://localhost:8000]
  --interface TEXT Network interface to use [default: wlan0]
  --help          Show help message and exit
```

## How It Works

### 1. Domain Resolution
The monitor resolves the target domain to IP addresses:

```python
def resolve_domain(self):
    """Resolve target domain to IP address"""
    try:
        result = socket.getaddrinfo(self.target_domain, None)
        ip_addresses = list(set([res[4][0] for res in result if ':' not in res[4][0]]))
        self.logger.info(f"Resolved {self.target_domain} to: {ip_addresses}")
        return ip_addresses[0] if ip_addresses else None
    except Exception as e:
        self.logger.error(f"Failed to resolve {self.target_domain}: {e}")
        return None
```

### 2. Traffic Generation
Generates HTTP requests to the target domain:

```python
async def generate_traffic(self):
    """Generate HTTP traffic to target domain"""
    try:
        async with aiohttp.ClientSession() as session:
            url = f"https://{self.target_domain}"
            async with session.get(url, timeout=10) as response:
                await response.text()
                self.logger.info(f"Generated traffic to {url} - Status: {response.status}")
    except Exception as e:
        self.logger.warning(f"Traffic generation failed: {e}")
```

### 3. Packet Capture
Captures network packets to/from the target IP:

```python
def start_packet_capture(self, target_ip):
    """Start packet capture for target IP"""
    packet_filter = f"host {target_ip}"
    
    def packet_handler(packet):
        if IP in packet:
            # Store packet data for analysis
            packet_data = {
                'timestamp': datetime.now(),
                'src_ip': packet[IP].src,
                'dst_ip': packet[IP].dst,
                'protocol': packet[IP].proto,
                'size': len(packet)
            }
            self.captured_packets.append(packet_data)
    
    # Start packet capture
    sniff(iface=self.interface, filter=packet_filter, 
          prn=packet_handler, timeout=self.capture_duration)
```

### 4. Traffic Analysis
Analyzes captured traffic using CICIDS2017 features:

```python
async def analyze_captured_traffic(self):
    """Analyze captured traffic for threats"""
    if not self.captured_packets:
        return
    
    # Group packets into flows
    flows = self.group_packets_into_flows()
    
    for flow_key, flow_data in flows.items():
        # Extract features using CICIDSFeatureExtractor
        features = self.feature_extractor.extract_features(None, flow_data)
        
        # Send to API for analysis
        try:
            prediction = await self.client.predict(features)
            if prediction['is_attack']:
                self.log_attack(prediction, flow_key)
        except Exception as e:
            self.logger.error(f"Analysis failed for flow {flow_key}: {e}")
```

## Monitoring Cycle

The website monitor operates in cycles:

### 1. Traffic Generation Phase
- Sends HTTP requests to target domain
- Uses various request types (GET, POST if applicable)
- Handles SSL/TLS connections

### 2. Capture Phase  
- Captures packets for configured duration (default: 5 seconds)
- Filters packets to target IP addresses only
- Stores packet metadata and payloads

### 3. Analysis Phase
- Groups packets into bidirectional flows
- Extracts CICIDS2017 features from flows
- Sends features to RL-IDS API for classification

### 4. Reporting Phase
- Logs detected attacks with full context
- Updates monitoring interface
- Prepares for next cycle

## Configuration

### Monitoring Parameters

```python
class WebsiteMonitor:
    def __init__(self, target_domain, api_url="http://localhost:8000", interface="wlan0"):
        # Traffic generation settings
        self.request_interval = 10  # seconds between requests
        self.capture_duration = 5   # seconds to capture after each request
```

### Customizable Settings

**Request Interval**: Time between traffic generation cycles
```python
self.request_interval = 10  # 10 seconds between requests
```

**Capture Duration**: How long to capture packets after each request
```python
self.capture_duration = 5   # 5 seconds of capture
```

**Network Interface**: Which interface to monitor
```python
self.interface = "wlan0"  # or eth0, etc.
```

## User Interface

The website monitor provides a real-time interface showing:

### Target Information
- Target domain name
- Resolved IP addresses
- Current monitoring status

### Traffic Statistics
- Requests generated
- Packets captured
- Flows analyzed
- Attacks detected

### Recent Activity
- Last request timestamp
- Recent packet captures
- Latest analysis results
- Detected threats

## Logging

### `website_monitor.log`
General monitoring activities:
```
2025-06-27 10:30:15 | INFO | Starting website monitoring for example.com
2025-06-27 10:30:16 | INFO | Resolved example.com to 93.184.216.34
2025-06-27 10:30:20 | INFO | Generated traffic to https://example.com - Status: 200
2025-06-27 10:30:25 | INFO | Captured 15 packets for analysis
```

### Alert Logs
Detected threats are logged with context:
```
2025-06-27 10:31:15 | WARNING | POTENTIAL THREAT DETECTED
  Target: example.com (93.184.216.34)
  Attack Type: DoS Hulk
  Confidence: 78.5%
  Flow: 192.168.1.100:45123 -> 93.184.216.34:443
```

### JSON Logs
Machine-readable monitoring data:
```json
{
  "timestamp": "2025-06-27T10:31:15.123456",
  "target_domain": "example.com",
  "target_ip": "93.184.216.34",
  "attack_detected": true,
  "attack_type": "DoS Hulk",
  "confidence": 0.785,
  "flow_details": {
    "src_ip": "192.168.1.100",
    "dst_ip": "93.184.216.34",
    "protocol": "TCP",
    "packets": 12,
    "bytes": 8456
  }
}
```

## Use Cases

### 1. Website Security Assessment
Monitor your own websites for attack patterns:
```bash
python website_monitor.py mywebsite.com
```

### 2. Third-party Service Monitoring
Analyze communication with external services:
```bash
python website_monitor.py api.thirdpartyservice.com
```

### 3. Suspicious Domain Investigation
Investigate potentially malicious domains:
```bash
python website_monitor.py suspicious-domain.example
```

### 4. Network Baseline Establishment
Establish normal communication patterns:
```bash
python website_monitor.py trusted-service.com
```

## Advanced Configuration

### Custom Request Headers
Modify traffic generation to include custom headers:

```python
headers = {
    'User-Agent': 'RL-IDS Website Monitor',
    'Accept': 'text/html,application/json',
    'Custom-Header': 'monitoring-traffic'
}
```

### Multiple Target IPs
Handle domains with multiple IP addresses:

```python
def resolve_all_ips(self, domain):
    """Resolve domain to all IP addresses"""
    result = socket.getaddrinfo(domain, None)
    return list(set([res[4][0] for res in result if ':' not in res[4][0]]))
```

### Protocol-Specific Monitoring
Monitor specific protocols beyond HTTP:

```python
# Monitor HTTPS traffic only
packet_filter = f"host {target_ip} and port 443"

# Monitor all traffic to domain
packet_filter = f"host {target_ip}"

# Monitor specific port ranges
packet_filter = f"host {target_ip} and portrange 80-443"
```

## Troubleshooting

### Common Issues

**Domain Resolution Failed**
```bash
# Check DNS resolution
nslookup example.com
dig example.com
```

**No Packets Captured**
- Verify network interface is correct
- Check if target domain is reachable
- Ensure sufficient privileges for packet capture

**High False Positive Rate**
- Adjust confidence threshold
- Add domain to whitelist if needed
- Verify baseline traffic patterns

**Connection Timeouts**
- Increase request timeout
- Check network connectivity
- Verify firewall rules

### Debug Mode

Enable verbose logging:
```bash
export RLIDS_DEBUG=true
export RLIDS_LOG_LEVEL=DEBUG
python website_monitor.py example.com
```

### Testing Network Connectivity

```bash
# Test basic connectivity
ping example.com

# Test HTTP connectivity
curl -I https://example.com

# Test with specific interface
curl --interface wlan0 https://example.com
```