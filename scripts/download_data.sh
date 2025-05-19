#!/usr/bin/env bash
set -eux

RAW_DIR="data/raw/CICIDS2017"
mkdir -p "$RAW_DIR"
cd "$RAW_DIR"

# Fetch the flow‑level CSV
CSV_URL="https://huggingface.co/datasets/c01dsnap/CIC-IDS2017/resolve/main/Wednesday-workingHours.pcap_ISCX.csv"
curl -L -o Wednesday-workingHours.pcap_ISCX.csv "$CSV_URL" 

echo "CSV saved to $PWD/Wednesday-workingHours.pcap_ISCX.csv"
