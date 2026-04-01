import urllib.request
import sys
import os

url = "https://ndownloader.figshare.com/files/37875672"
filename = "Train.zip"

print(f"Downloading {filename} directly via Python to bypass curl/wget issues...")

try:
    # Open the URL
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req) as response:
        # Get total size
        file_size = int(response.info().get('Content-Length', -1))
        
        if file_size == -1:
            print("Warning: Could not determine file size.")
            
        # Download in chunks
        chunk_size = 8192 * 4  # 32KB chunks
        downloaded = 0
        
        with open(filename, 'wb') as f_out:
            while True:
                buffer = response.read(chunk_size)
                if not buffer:
                    break
                f_out.write(buffer)
                downloaded += len(buffer)
                
                # Print progress
                if file_size > 0:
                    percent = downloaded / file_size * 100
                    sys.stdout.write(f"\rProgress: {downloaded / (1024*1024):.2f} MB / {file_size / (1024*1024):.2f} MB ({percent:.1f}%)")
                    sys.stdout.flush()
                else:
                    sys.stdout.write(f"\rDownloaded: {downloaded / (1024*1024):.2f} MB")
                    sys.stdout.flush()
                    
    print("\n\n✅ Download complete!")
    
except Exception as e:
    print(f"\n❌ Error downloading: {e}")
