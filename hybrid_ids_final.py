import joblib
import pandas as pd
from scapy.all import sniff, IP, TCP, ICMP
import warnings

# Ignore those annoying 'feature names' warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Load the brain
print("Loading Optimized ML Model Brain...")
model = joblib.load('optimized_ids_model.pkl')

def hybrid_detection(packet):
    if packet.haslayer(IP):
        ip_src = packet[IP].src

        # --- 1. RULE-BASED LAYER (Fast Check) ---
        if packet.haslayer(ICMP):
            print(f"[!] RULE ALERT: Ping/ICMP detected from {ip_src}")
            return

        # --- 2. MACHINE LEARNING LAYER (Smart Check) ---
        if packet.haslayer(TCP):
            # Extract features exactly as the model expects
            dest_port = packet[TCP].dport
            features = [[dest_port, 100, 1, 1]]
            # Use probability for higher sensitivity
            proba = model.predict_proba(features)[0][1]

            # If the probability is high, it's an attack
            if proba > 0.5:
                print(f"[!!!] ML ALERT: Malicious Pattern (Confidence: {proba:.2f}) from {ip_src}")
            else:
                # This helps you see that it's actually working
                print(f"[-] Normal TCP traffic from {ip_src} (Safe: {1-proba:.2f})")

print("--- HYBRID IDS: MONITORING ACTIVE ---")
# Start sniffing - using 'lo' for localhost testing
sniff(iface="lo", prn=hybrid_detection, store=0)
