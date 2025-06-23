import asyncio
import scapy.all as scapy
from scapy.layers.inet import IP, TCP, UDP, ICMP
from scapy.layers.l2 import Ether
import pandas as pd
import numpy as np
import json
import time
import logging
import threading
from collections import defaultdict, deque
from datetime import datetime, timedelta
import subprocess
import psutil
import os
from api.client import IDSAPIClient
from concurrent.futures import ThreadPoolExecutor
import queue

class SimpleTerminalInterface:
    """Simple terminal interface for RL-IDS Monitor"""
    
    def __init__(self):
        self.console_width = 80
        
    def clear_screen(self):
        """Clear terminal screen"""
        os.system('clear' if os.name == 'posix' else 'cls')
    
    def print_header(self):
        """Print header"""
        print("=" * self.console_width)
        print("🛡️  RL-IDS NETWORK MONITOR 🛡️".center(self.console_width))
        print("Real-time Intrusion Detection System".center(self.console_width))
        print("=" * self.console_width)
    
    def print_stats(self, stats, interface_info, alerts):
        """Print current statistics"""
        self.clear_screen()
        self.print_header()
        
        # Calculate uptime and rate
        uptime = datetime.now() - stats['start_time']
        uptime_str = str(uptime).split('.')[0]  # Remove microseconds
        packets_per_min = stats['packets_analyzed'] / max(uptime.total_seconds() / 60, 1)
        
        # Statistics section
        print("\n📈 STATISTICS:")
        print(f"  📊 Uptime:        {uptime_str}")
        print(f"  📦 Packets:       {stats['packets_analyzed']:,}")
        print(f"  ⚡ Rate:          {packets_per_min:.1f}/min")
        print(f"  🔍 Active Flows:  {stats['active_flows']}")
        print(f"  🚨 Attacks:       {stats['attacks_detected']}")
        print(f"  🔇 Ignored:       {stats['attacks_ignored']}")
        print(f"  📤 Queue Size:    {stats['queue_size']}")
        
        # Configuration section
        print("\n🔧 CONFIGURATION:")
        print(f"  📡 Interface:     {interface_info['name']}")
        print(f"  🌐 API URL:       {interface_info['api_url']}")
        print(f"  ⚙️ Threshold:     {interface_info['threshold']:.1%}")
        print(f"  🔧 Status:        {interface_info['status']}")
        
        # Recent alerts section
        print("\n🚨 RECENT ALERTS:")
        if not alerts:
            print("  🟢 No recent alerts")
        else:
            print("  Time     | Attack Type      | Source IP       | Confidence")
            print("  " + "-" * 65)
            for alert in alerts[-10:]:  # Show last 10 alerts
                time_str = alert['time'].ljust(8)
                attack_str = alert['attack_type'][:15].ljust(15)
                source_str = alert['source_ip'][:15].ljust(15)
                conf_str = f"{alert['confidence']:.0%}".rjust(10)
                print(f"  {time_str} | {attack_str} | {source_str} | {conf_str}")
        
        print("\n" + "=" * self.console_width)
        print("Press Ctrl+C to stop monitoring")
        print("Logs saved to: logs/")
    
    def show_startup_banner(self, interface, api_url):
        """Show startup banner"""
        self.clear_screen()
        self.print_header()
        print()
        print(f"  Interface: {interface}")
        print(f"  API URL:   {api_url}")
        print(f"  Status:    Starting...")
        print()
        print("Initializing...")
        time.sleep(2)
    
    def print_error(self, message):
        """Print error message"""
        print(f"❌ ERROR: {message}")
    
    def print_success(self, message):
        """Print success message"""
        print(f"✅ {message}")
    
    def print_warning(self, message):
        """Print warning message"""
        print(f"⚠️  WARNING: {message}")


class RealTimeNetworkMonitor:
    """Real-time network traffic monitor using RL-IDS API"""
    
    def __init__(self, interface="wlan0", api_url="http://localhost:8000", target_domain=None):
        self.interface = interface
        self.api_url = api_url
        self.client = IDSAPIClient(api_url)
        
        # Simple UI
        self.ui = SimpleTerminalInterface()
        
        # Flow tracking for statistical features
        self.flows = defaultdict(lambda: {
            'packets': [],
            'start_time': None,
            'last_seen': None,
            'forward_packets': 0,
            'backward_packets': 0,
            'forward_bytes': 0,
            'backward_bytes': 0,
            'tcp_flags': []
        })
        
        # Packet buffers for time-window analysis
        self.packet_buffer = deque(maxlen=1000)
        self.alert_log = []
        
        # Configuration
        self.confidence_threshold = 0.7
        self.monitoring_window = 10  # seconds
        self.cleanup_interval = 300  # 5 minutes
        
        # Attack types to ignore
        self.ignored_attacks = ['heartbleed', 'portscan']
        
        # Add packet queue for async processing
        self.packet_queue = queue.Queue(maxsize=1000)
        self.processing_active = True
        
        self.setup_logging()
        self.feature_extractor = CICIDSFeatureExtractor()
        
        # Statistics
        self.stats = {
            'packets_analyzed': 0,
            'attacks_detected': 0,
            'attacks_ignored': 0,
            'last_attack': None,
            'start_time': datetime.now(),
            'active_flows': 0,
            'queue_size': 0
        }
        
        # Interface info for UI
        self.interface_info = {
            'name': interface,
            'api_url': api_url,
            'threshold': self.confidence_threshold,
            'status': 'Initializing'
        }
        
        self.target_domain = target_domain
        self.target_ips = set()
        
        # Resolve domain to IPs if provided
        if self.target_domain:
            self.resolve_target_domain()
    
    def resolve_target_domain(self):
        """Resolve target domain to IP addresses"""
        import socket
        try:
            # Remove https:// if present
            domain = self.target_domain.replace('https://', '').replace('http://', '')
            domain = domain.split('/')[0]  # Remove path if present
            
            # Resolve domain to IP
            ip = socket.gethostbyname(domain)
            self.target_ips.add(ip)
            self.ui.print_success(f"Resolved {domain} to {ip}")
            
        except socket.gaierror as e:
            self.ui.print_error(f"Failed to resolve {domain}: {e}")
    
    def is_target_traffic(self, packet):
        """Check if packet is related to target domain"""
        if not self.target_domain or not self.target_ips:
            return True  # Monitor all traffic if no target specified
        
        if IP not in packet:
            return False
        
        src_ip = packet[IP].src
        dst_ip = packet[IP].dst
        
        return src_ip in self.target_ips or dst_ip in self.target_ips
    
    def setup_logging(self):
        """Setup production logging"""
        os.makedirs('logs', exist_ok=True)
        
        # Configure logging to file only
        logging.basicConfig(
            level=logging.WARNING,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('logs/network_monitor.log'),
                logging.FileHandler('logs/attacks.log', mode='a'),
            ]
        )
        
        # Silence noisy loggers
        logging.getLogger('httpx').setLevel(logging.WARNING)
        logging.getLogger('scapy').setLevel(logging.WARNING)
        
        self.logger = logging.getLogger('NetworkMonitor')
        
        # Attack logger
        self.attack_logger = logging.getLogger('AttackLogger')
        attack_handler = logging.FileHandler('logs/intrusion_alerts.log')
        attack_handler.setFormatter(logging.Formatter(
            '%(asctime)s - ATTACK - %(message)s'
        ))
        self.attack_logger.addHandler(attack_handler)
        self.attack_logger.setLevel(logging.WARNING)
    
    def get_available_interfaces(self):
        """Get list of available network interfaces"""
        interfaces = []
        for interface, addrs in psutil.net_if_addrs().items():
            if any(addr.family.name == 'AF_INET' for addr in addrs):
                interfaces.append(interface)
        return interfaces
    
    def start_monitoring(self):
        """Start network monitoring"""
        # Show startup banner
        self.ui.show_startup_banner(self.interface, self.api_url)
        
        # Check if interface exists
        available_interfaces = self.get_available_interfaces()
        if self.interface not in available_interfaces:
            self.ui.print_error(f"Interface {self.interface} not found!")
            print(f"Available interfaces: {available_interfaces}")
            return
        
        # Test API connection
        try:
            asyncio.run(self.test_api_connection())
            self.interface_info['status'] = 'API Connected ✅'
            self.ui.print_success("API connection successful")
        except Exception as e:
            self.ui.print_error(f"Cannot connect to API: {e}")
            return
        
        # Start background threads
        self.ui.print_success("Starting background processes...")
        
        # Start async packet processor
        processor_thread = threading.Thread(target=self.start_async_processor, daemon=True)
        processor_thread.start()
        
        # Start cleanup thread
        cleanup_thread = threading.Thread(target=self.cleanup_old_flows, daemon=True)
        cleanup_thread.start()
        
        # Start UI update thread
        stats_thread = threading.Thread(target=self.update_ui_stats, daemon=True)
        stats_thread.start()
        
        self.interface_info['status'] = 'Monitoring Active 🟢'
        
        # Start packet capture
        self.ui.print_success(f"Starting packet capture on {self.interface}...")
        time.sleep(1)
        
        # Create filter string
        packet_filter = "ip"
        if self.target_ips:
            ip_filters = " or ".join([f"host {ip}" for ip in self.target_ips])
            packet_filter = f"ip and ({ip_filters})"
        
        try:
            scapy.sniff(
                iface=self.interface,
                prn=self.process_packet,
                store=False,
                filter=packet_filter,  # Updated filter
                stop_filter=lambda x: False
            )
        except PermissionError:
            self.ui.print_error("Permission denied! Run with sudo")
        except Exception as e:
            self.ui.print_error(f"Packet capture error: {e}")
            import traceback
            print(f"Traceback: {traceback.format_exc()}")
        finally:
            self.processing_active = False
    
    def start_async_processor(self):
        """Start async packet processor in separate thread"""
        try:
            asyncio.run(self.packet_processor_loop())
        except Exception as e:
            self.logger.error(f"Error in async processor: {e}")
    
    async def packet_processor_loop(self):
        """Async loop to process packets from queue"""
        while self.processing_active or not self.packet_queue.empty():
            try:
                try:
                    packet_data = self.packet_queue.get(timeout=1)
                except queue.Empty:
                    continue
                
                await self.process_packet_async(packet_data)
                self.packet_queue.task_done()
                
            except Exception as e:
                self.logger.error(f"Error in packet processor loop: {e}")
                await asyncio.sleep(0.1)
    
    async def test_api_connection(self):
        """Test connection to RL-IDS API"""
        try:
            health = await self.client.health_check()
            return health
        except Exception as e:
            raise Exception(f"API connection failed: {e}")
    
    def process_packet(self, packet):
        """Process each captured packet"""
        try:
            if IP not in packet:
                return
            
            # Filter by target domain if specified
            if not self.is_target_traffic(packet):
                return
                
            self.stats['packets_analyzed'] += 1
            
            packet_info = {
                'timestamp': datetime.now(),
                'packet': packet,
                'size': len(packet)
            }
            self.packet_buffer.append(packet_info)
            
            flow_key = self.get_flow_key(packet)
            if flow_key:
                self.update_flow_stats(flow_key, packet)
                
                features = self.feature_extractor.extract_features(packet, self.flows[flow_key])
                if features and len(features) == 78:
                    packet_data = {
                        'packet': packet,
                        'features': features,
                        'flow_key': flow_key,
                        'timestamp': datetime.now()
                    }
                    
                    try:
                        self.packet_queue.put_nowait(packet_data)
                        self.stats['queue_size'] = self.packet_queue.qsize()
                    except queue.Full:
                        pass  # Silently drop if queue full
                    
        except Exception as e:
            self.logger.error(f"Error processing packet: {e}")
    
    async def process_packet_async(self, packet_data):
        """Process packet data asynchronously"""
        try:
            packet = packet_data['packet']
            features = packet_data['features']
            flow_key = packet_data['flow_key']
            
            # Analyze with RL-IDS API
            prediction = await self.client.predict(features)
            
            # Create analysis result
            result = {
                'timestamp': packet_data['timestamp'].isoformat(),
                'flow_key': str(flow_key),
                'src_ip': packet[IP].src,
                'dst_ip': packet[IP].dst,
                'protocol': packet[IP].proto,
                'prediction': prediction
            }
            
            # Check if attack detected
            if prediction['is_attack'] and prediction['confidence'] >= self.confidence_threshold:
                attack_type = prediction.get('predicted_class', '')
                
                if self.is_attack_ignored(attack_type):
                    self.stats['attacks_ignored'] += 1
                    # Log to file but don't show in UI
                    with open('logs/ignored_attacks.json', 'a') as f:
                        ignored_alert = {
                            'timestamp': result['timestamp'],
                            'attack_type': attack_type,
                            'source_ip': result['src_ip'],
                            'confidence': prediction['confidence'],
                            'reason': 'Attack type in ignore list'
                        }
                        f.write(json.dumps(ignored_alert) + ',\n')
                else:
                    await self.handle_attack_detection(result, packet)
            
        except Exception as e:
            self.logger.error(f"Error analyzing packet: {e}")
    
    def is_attack_ignored(self, attack_type):
        """Check if attack type should be ignored"""
        attack_lower = attack_type.lower().strip()
        return any(ignored.lower() in attack_lower for ignored in self.ignored_attacks)
    
    def get_flow_key(self, packet):
        """Generate unique flow identifier"""
        if IP not in packet:
            return None
            
        ip = packet[IP]
        src_ip, dst_ip = ip.src, ip.dst
        proto = ip.proto
        
        src_port = dst_port = 0
        if TCP in packet:
            src_port, dst_port = packet[TCP].sport, packet[TCP].dport
        elif UDP in packet:
            src_port, dst_port = packet[UDP].sport, packet[UDP].dport
        
        if (src_ip, src_port) < (dst_ip, dst_port):
            return (src_ip, dst_ip, src_port, dst_port, proto)
        else:
            return (dst_ip, src_ip, dst_port, src_port, proto)
    
    def update_flow_stats(self, flow_key, packet):
        """Update flow statistics for feature extraction"""
        flow = self.flows[flow_key]
        current_time = datetime.now()
        
        if flow['start_time'] is None:
            flow['start_time'] = current_time
        
        flow['last_seen'] = current_time
        flow['packets'].append({
            'timestamp': current_time,
            'size': len(packet),
            'direction': self.get_packet_direction(packet, flow_key)
        })
        
        packet_size = len(packet)
        direction = self.get_packet_direction(packet, flow_key)
        
        if direction == 'forward':
            flow['forward_packets'] += 1
            flow['forward_bytes'] += packet_size
        else:
            flow['backward_packets'] += 1
            flow['backward_bytes'] += packet_size
        
        if TCP in packet:
            flow['tcp_flags'].append(packet[TCP].flags)
    
    def get_packet_direction(self, packet, flow_key):
        """Determine packet direction in flow"""
        ip = packet[IP]
        if ip.src == flow_key[0]:
            return 'forward'
        else:
            return 'backward'
    
    async def handle_attack_detection(self, analysis_result, packet):
        """Handle detected attack"""
        prediction = analysis_result['prediction']
        
        self.stats['attacks_detected'] += 1
        self.stats['last_attack'] = datetime.now()
        
        # Create UI-friendly alert
        alert = {
            'time': datetime.now().strftime('%H:%M:%S'),
            'attack_type': prediction['predicted_class'],
            'source_ip': analysis_result['src_ip'],
            'confidence': prediction['confidence'],
            'severity': self.get_severity_level(prediction['confidence'])
        }
        
        # Add to UI alerts
        self.alert_log.append(alert)
        
        # Create detailed alert for logging
        detailed_alert = {
            'alert_id': len(self.alert_log),
            'timestamp': analysis_result['timestamp'],
            'severity': alert['severity'],
            'attack_type': alert['attack_type'],
            'confidence': alert['confidence'],
            'source_ip': alert['source_ip'],
            'destination_ip': analysis_result['dst_ip'],
            'protocol': self.get_protocol_name(analysis_result['protocol']),
            'flow_key': analysis_result['flow_key'],
            'packet_info': self.get_packet_details_serializable(packet),
            'recommended_action': self.get_recommended_action(prediction)
        }
        
        # Log to file
        self.attack_logger.warning(f"INTRUSION DETECTED: {json.dumps(detailed_alert, indent=2)}")
        
        # Save to alerts file
        with open('logs/alerts.json', 'a') as f:
            f.write(json.dumps(detailed_alert) + ',\n')
        
        # Take defensive action if needed
        if alert['severity'] in ['CRITICAL', 'HIGH']:
            await self.take_defensive_action(detailed_alert)
    
    def get_packet_details_serializable(self, packet):
        """Extract packet details in JSON-serializable format"""
        details = {
            'packet_size': len(packet),
            'ttl': packet[IP].ttl if IP in packet else None,
        }
        
        if TCP in packet:
            tcp = packet[TCP]
            details.update({
                'src_port': tcp.sport,
                'dst_port': tcp.dport,
                'tcp_flags': int(tcp.flags),
                'window_size': tcp.window,
                'seq_num': tcp.seq,
                'ack_num': tcp.ack
            })
        elif UDP in packet:
            udp = packet[UDP]
            details.update({
                'src_port': udp.sport,
                'dst_port': udp.dport,
                'udp_length': udp.len
            })
        
        return details
    
    def get_severity_level(self, confidence):
        """Determine severity level based on confidence"""
        if confidence >= 0.95:
            return "CRITICAL"
        elif confidence >= 0.85:
            return "HIGH"
        elif confidence >= 0.7:
            return "MEDIUM"
        else:
            return "LOW"
    
    def get_protocol_name(self, proto_num):
        """Convert protocol number to name"""
        protocols = {1: 'ICMP', 6: 'TCP', 17: 'UDP'}
        return protocols.get(proto_num, f'PROTO_{proto_num}')
    
    def get_recommended_action(self, prediction):
        """Get recommended action based on prediction"""
        attack_type = prediction['predicted_class']
        confidence = prediction['confidence']
        
        if confidence >= 0.9:
            return f"IMMEDIATE: Block {attack_type} traffic and investigate"
        elif confidence >= 0.7:
            return f"MONITOR: Increase monitoring for {attack_type} patterns"
        else:
            return f"LOG: Continue monitoring {attack_type} activity"
    
    async def take_defensive_action(self, alert):
        """Take automated defensive actions"""
        try:
            src_ip = alert['source_ip']
            if alert['severity'] == 'CRITICAL':
                await self.block_ip_iptables(src_ip)
        except Exception as e:
            self.logger.error(f"Error taking defensive action: {e}")
    
    async def block_ip_iptables(self, ip_address):
        """Block IP using iptables"""
        try:
            check_cmd = f"sudo iptables -L INPUT -n | grep {ip_address}"
            result = subprocess.run(check_cmd, shell=True, capture_output=True)
            
            if result.returncode != 0:
                block_cmd = f"sudo iptables -A INPUT -s {ip_address} -j DROP"
                subprocess.run(block_cmd, shell=True, check=True)
                self.logger.warning(f"BLOCKED IP: {ip_address}")
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Failed to block IP {ip_address}: {e}")
    
    def cleanup_old_flows(self):
        """Cleanup old flow entries periodically"""
        while True:
            try:
                current_time = datetime.now()
                cutoff_time = current_time - timedelta(seconds=self.cleanup_interval)
                
                old_flows = [
                    flow_key for flow_key, flow_data in self.flows.items()
                    if flow_data.get('last_seen', current_time) < cutoff_time
                ]
                
                for flow_key in old_flows:
                    del self.flows[flow_key]
                
                self.stats['active_flows'] = len(self.flows)
                
            except Exception as e:
                self.logger.error(f"Error in cleanup: {e}")
            
            time.sleep(self.cleanup_interval)
    
    def update_ui_stats(self):
        """Update UI statistics periodically"""
        while True:
            try:
                time.sleep(2)  # Update every 2 seconds
                
                self.stats['active_flows'] = len(self.flows)
                self.stats['queue_size'] = self.packet_queue.qsize()
                
                # Update UI display
                self.ui.print_stats(
                    self.stats,
                    self.interface_info,
                    self.alert_log
                )
                
            except Exception as e:
                self.logger.error(f"Error updating UI: {e}")


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


# Usage
if __name__ == "__main__":
    import sys
    import argparse
    
    parser = argparse.ArgumentParser(description='RL-IDS Network Monitor')
    parser.add_argument('interface', nargs='?', help='Network interface to monitor')
    parser.add_argument('--target', help='Target domain to monitor (e.g., https://www.example.com)')
    parser.add_argument('--api-url', default='http://localhost:8000', help='RL-IDS API URL')
    
    args = parser.parse_args()
    
    # Get network interface
    if args.interface:
        interface = args.interface
    else:
        # Auto-detect interface
        interfaces = []
        preferred_interfaces = []
        
        for iface, addrs in psutil.net_if_addrs().items():
            if any(addr.family.name == 'AF_INET' for addr in addrs):
                interfaces.append(iface)
                if not iface.startswith(('lo', 'docker', 'br-', 'veth')):
                    preferred_interfaces.append(iface)
        
        if preferred_interfaces:
            interface = preferred_interfaces[0]
        elif interfaces:
            interface = interfaces[0]
        else:
            print("No network interfaces found!")
            sys.exit(1)
    
    # Start monitoring
    monitor = RealTimeNetworkMonitor(
        interface=interface,
        api_url=args.api_url,
        target_domain=args.target
    )
    
    try:
        monitor.start_monitoring()
    except KeyboardInterrupt:
        print("\n🛑 Monitoring stopped by user")
    except Exception as e:
        print(f"❌ Monitor error: {e}")
        import traceback
        print(f"Full traceback: {traceback.format_exc()}")