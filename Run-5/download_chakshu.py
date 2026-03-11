#!/usr/bin/env python3
"""
CHAKSHU Dataset Downloader - Uses Figshare API
Downloads all files from the dataset article
"""
import urllib.request
import json
import os

article_id = 20123135
url = f"https://api.figshare.com/v2/articles/{article_id}/files"

print("=" * 60)
print("   CHAKSHU DATASET DOWNLOADER")
print("=" * 60)
print(f"\nFetching file list for Article {article_id}...")

try:
    # Get the file list from Figshare API
    req = urllib.request.Request(url)
    response = urllib.request.urlopen(req)
    data = json.loads(response.read().decode())
    
    print(f"Found {len(data)} files. Starting downloads...\n")
    
    for f in data:
        name = f['name']
        download_url = f['download_url']
        size_mb = f['size'] / (1024 * 1024)
        
        print(f"Downloading: {name} ({size_mb:.2f} MB)")
        # Use wget to download (quoted to handle spaces safely)
        os.system(f"wget -q --show-progress -O '{name}' '{download_url}'")
        print("Done.\n")
    
    print("=" * 60)
    print("✅ All files downloaded successfully!")
    print("=" * 60)
        
except Exception as e:
    print(f"\n❌ Error: {e}")
    print("\nPlease download manually from:")
    print("https://doi.org/10.6084/m9.figshare.11857698.v2")
