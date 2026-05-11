import subprocess
import wget
from pathlib import Path
import os

def download_video(video_url):
    os.makedirs("videos", exist_ok=True)
    input_path = wget.download(video_url, out="videos/")
    
    output_path = Path(input_path).parent / (str(Path(input_path).stem) + '.mp4')
    convert_flv_to_mp4(input_path, output_path)
    return output_path

def convert_flv_to_mp4(input_path, output_path):
    """
    Convert an FLV file to MP4 using FFmpeg.

    :param input_path: Path to the input FLV file.
    :param output_path: Path to the output MP4 file.
    """
    command = [
        "ffmpeg",
        "-i", input_path,    # Input file
        "-c:v", "copy",      # Copy video stream
        "-c:a", "copy",      # Copy audio stream
        output_path          # Output file
    ]
    process = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if process.returncode == 0:
        print(f"Conversion successful: {output_path}")
    else:
        print(f"Error during conversion: {process.stderr.decode()}")