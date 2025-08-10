import os

VIDEO_URL = "https://www.youtube.com/watch?v=MNn9qKG2UFI"
OUTPUT_FILE = "input-video.mp4"

def download_video(url, output_file):
    # Ensure yt-dlp is installed
    os.system(f'yt-dlp -f "bestvideo[height=2160]+bestaudio/best" --merge-output-format mp4 -o "{output_file}" {url}')

if __name__ == "__main__":
    download_video(VIDEO_URL, OUTPUT_FILE)
