import asyncio
import requests
import time
import threading
from datetime import datetime
import json
import subprocess
import os
from api.client import IDSAPIClient
from network_monitor import SimpleTerminalInterface, CICIDSFeatureExtractor
import scapy.all as scapy
from scapy.layers.inet import IP, TCP
import psutil

class WebsiteMonitor:
    """Monitor external websites by generating traffic and analyzing responses"""
    
    def __init__(self, target_domain, api_url="http://localhost:8000", interface="wlan0"):
        self.target_domain = target_domain.replace('https://', '').replace('http://', '')
        self.api_url = api_url
        self.interface = interface
        self.client = IDSAPIClient(api_url)
        self.ui = SimpleTerminalInterface()
        
        # Traffic generation settings
        self.request_interval = 10  # seconds
        self.capture_duration = 5   # seconds to capture after each request
        
        # Packet capture
        self.captured_packets = []
        self.capturing = False
        
        # Statistics
        self.stats = {
            'requests_sent': 0,
            'packets_captured': 0,
            'attacks_detected': 0,
            'start_time': datetime.now(),
        }
        
        # Setup
        self.setup_logging()
        self.feature_extractor = CICIDSFeatureExtractor()
        
    def setup_logging(self):
        """Setup logging"""
        os.makedirs('logs', exist_ok=True)
        import logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('logs/website_monitor.log'),
            ]
        )
        self.logger = logging.getLogger('WebsiteMonitor')
    
    async def start_monitoring(self):
        """Start website monitoring"""
        self.ui.show_startup_banner(f"Website: {self.target_domain}", self.api_url)
        
        # Test API connection
        try:
            await self.client.health_check()
            self.ui.print_success("API connection successful")
        except Exception as e:
            self.ui.print_error(f"Cannot connect to API: {e}")
            return
        
        # Resolve domain
        target_ip = self.resolve_domain()
        if not target_ip:
            return
        
        self.ui.print_success(f"Resolved {self.target_domain} to {target_ip}")
        
        # Start monitoring loop
        try:
            while True:
                await self.monitor_cycle(target_ip)
                await asyncio.sleep(self.request_interval)
        except KeyboardInterrupt:
            self.ui.print_success("Monitoring stopped by user")
        finally:
            await self.client.close()
    
    def resolve_domain(self):
        """Resolve domain to IP"""
        import socket
        try:
            return socket.gethostbyname(self.target_domain)
        except socket.gaierror as e:
            self.ui.print_error(f"Failed to resolve {self.target_domain}: {e}")
            return None
    
    async def monitor_cycle(self, target_ip):
        """Complete monitoring cycle: generate traffic + capture + analyze"""
        # Start packet capture
        capture_thread = threading.Thread(
            target=self.start_packet_capture, 
            args=(target_ip,), 
            daemon=True
        )
        capture_thread.start()
        
        # Wait a moment for capture to start
        await asyncio.sleep(0.5)
        
        # Generate traffic
        await self.generate_traffic()
        
        # Wait for capture to complete
        await asyncio.sleep(self.capture_duration)
        self.capturing = False
        capture_thread.join(timeout=2)
        
        # Analyze captured packets
        await self.analyze_captured_traffic()
        
        # Update UI
        self.update_ui()
    
    async def generate_traffic(self):
        """Generate HTTP traffic to target"""
        base_url = f"https://{self.target_domain}"
        
        patterns = [
            '/api/health',
            '/',
            '/about',
            '/contact',
            '/api/data',
            '/admin',  # This might trigger security responses
            '/wp-admin', # Common attack vector
        ]
        
        for path in patterns:
            try:
                url = base_url + path
                response = requests.get(url, timeout=5, allow_redirects=True)
                self.stats['requests_sent'] += 1
                await asyncio.sleep(0.2)  # Small delay between requests
            except requests.RequestException:
                pass  # Expected for some URLs
    
    def start_packet_capture(self, target_ip):
        """Capture packets related to target IP"""
        self.capturing = True
        self.captured_packets = []
        
        def packet_handler(packet):
            if not self.capturing:
                return True  # Stop capture
            
            if IP in packet:
                if packet[IP].src == target_ip or packet[IP].dst == target_ip:
                    self.captured_packets.append(packet)
                    self.stats['packets_captured'] += 1
        
        try:
            scapy.sniff(
                iface=self.interface,
                prn=packet_handler,
                filter=f"host {target_ip}",
                timeout=self.capture_duration + 1,
                stop_filter=lambda x: not self.capturing
            )
        except Exception as e:
            self.logger.error(f"Packet capture error: {e}")
    
    async def analyze_captured_traffic(self):
        """Analyze captured packets for threats"""
        if not self.captured_packets:
            return
        
        # Group packets by flow
        flows = {}
        
        for packet in self.captured_packets:
            flow_key = self.get_flow_key(packet)
            if flow_key not in flows:
                flows[flow_key] = {
                    'packets': [],
                    'start_time': datetime.now(),
                    'forward_packets': 0,
                    'backward_packets': 0,
                    'forward_bytes': 0,
                    'backward_bytes': 0,
                    'tcp_flags': []
                }
            
            flows[flow_key]['packets'].append({
                'timestamp': datetime.now(),
                'size': len(packet),
                'direction': 'forward'  # Simplified
            })
            
            if TCP in packet:
                flows[flow_key]['tcp_flags'].append(packet[TCP].flags)
        
        # Analyze each flow
        for flow_key, flow_data in flows.items():
            features = self.feature_extractor.extract_features(
                self.captured_packets[0], flow_data
            )
            
            if features and len(features) == 78:
                try:
                    prediction = await self.client.predict(features)
                    
                    if prediction['is_attack'] and prediction['confidence'] > 0.7:
                        self.stats['attacks_detected'] += 1
                        self.log_attack(prediction, flow_key)
                        
                except Exception as e:
                    self.logger.error(f"Prediction error: {e}")
    
    def get_flow_key(self, packet):
        """Get flow identifier"""
        if IP not in packet:
            return None
        
        ip = packet[IP]
        src_port = dst_port = 0
        
        if TCP in packet:
            src_port, dst_port = packet[TCP].sport, packet[TCP].dport
        
        return (ip.src, ip.dst, src_port, dst_port, ip.proto)
    
    def log_attack(self, prediction, flow_key):
        """Log detected attack"""
        alert = {
            'timestamp': datetime.now().isoformat(),
            'target_domain': self.target_domain,
            'flow_key': str(flow_key),
            'attack_type': prediction['predicted_class'],
            'confidence': prediction['confidence'],
            'is_attack': prediction['is_attack']
        }
        
        with open('logs/website_attacks.json', 'a') as f:
            f.write(json.dumps(alert) + '\n')
        
        self.logger.warning(f"ATTACK DETECTED: {alert}")
    
    def update_ui(self):
        """Update UI with current stats"""
        self.ui.clear_screen()
        self.ui.print_header()
        
        uptime = datetime.now() - self.stats['start_time']
        
        print(f"\n🌐 TARGET: {self.target_domain}")
        print(f"📊 Uptime: {str(uptime).split('.')[0]}")
        print(f"📤 Requests: {self.stats['requests_sent']}")
        print(f"📦 Packets: {self.stats['packets_captured']}")
        print(f"🚨 Attacks: {self.stats['attacks_detected']}")
        print(f"\nPress Ctrl+C to stop...")


async def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Website Monitor for RL-IDS')
    parser.add_argument('target', help='Target website (e.g., yashpotdar.vercel.app)')
    parser.add_argument('--api-url', default='http://localhost:8000', help='RL-IDS API URL')
    parser.add_argument('--interface', default='wlan0', help='Network interface')
    
    args = parser.parse_args()
    
    monitor = WebsiteMonitor(
        target_domain=args.target,
        api_url=args.api_url,
        interface=args.interface
    )
    
    await monitor.start_monitoring()

if __name__ == "__main__":
    asyncio.run(main())