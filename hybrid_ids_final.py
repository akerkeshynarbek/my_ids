import joblib
import pandas as pd
from scapy.all import sniff, IP, TCP, ICMP
import warnings

# Ignoring  not needed warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Loading the brain
print("Loading Optimized ML Model Brain...")
model = joblib.load('optimized_ids_model.pkl')

def hybrid_detection(packet):
    if packet.haslayer(IP):
        ip_src = packet[IP].src

        #RULE-BASED LAYER
        if packet.haslayer(ICMP):
            print(f"[!] RULE ALERT: Ping/ICMP detected from {ip_src}")
            return

        #MACHINE LEARNING LAYER
        if packet.haslayer(TCP):
            # Extracting features as the model expects
            dest_port = packet[TCP].dport
            features = [[dest_port, 100, 1, 1]]
            #probability for higher sensitivity
            proba = model.predict_proba(features)[0][1]

            # if the probability is high, i consider it as an attack
            if proba > 0.5:
                print(f"[!!!] ML ALERT: Malicious Pattern (Confidence: {proba:.2f}) from {ip_src}")
            else:
                # This helps me see that it's actually working
                print(f"[-] Normal TCP traffic from {ip_src} (Safe: {1-proba:.2f})")

print("--- HYBRID IDS: MONITORING ACTIVE ---")
# Starting sniffing
sniff(iface="lo", prn=hybrid_detection, store=0)
