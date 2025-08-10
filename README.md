# 🚦 Traffic Flow Analysis

This project analyzes traffic flow from a video feed using **YOLO object detection** and **object tracking**.  
It detects vehicles, assigns them to lanes, counts them as they pass a predefined line, and optionally logs data to a CSV file.  

---

## 📂 Project Structure

```
├── TrafficAnalysis.ipynb # Main Jupyter Notebook for traffic detection, tracking & analysis
├── video_download.py # Script to download 4K demo traffic video from YouTube
├── input-video.mp4 # Downloaded video (git-ignored if large)
├── requirements.txt # Python dependencies
├── laneCount.csv # Csv file for logging
└── README.md # Project documentation
```

---

## 📜 Features
- **YOLO-based vehicle detection** for accurate recognition.
- **Object tracking** using ByteTrack.
- **Lane-wise vehicle counting** with a horizontal counting line.
- **CSV logging** for lane-wise and total counts.
- **Supports 4K demo video download** from YouTube.
- **Customizable lanes & counting line** positions.

---
## 📹 Demo Output Video
[Google Drive Link..]()

## 📹 Demo Video
By default, the project uses a sample 4K traffic video from YouTube:  
`https://www.youtube.com/watch?v=MNn9qKG2UFI`

To download it:
```bash
  python video_download.py
```
This will save it as `input-video.mp4` for use in the notebook.

---
## ⚙️ Installation
- **Clone this repository**

```bash
    git clone https://github.com/Kashi-onGit/traffic-flow-analysis.git
    cd traffic-flow-analysis
```

- **Install dependencies**

```bash
  pip install -r requirements.txt
```

- **(Optional) Install FFmpeg** – Required for merging audio & video from YouTube downloads.
  - Windows:[ Download FFmpeg ](https://www.gyan.dev/ffmpeg/builds/)
  - Linux/macOS:
```bash
  sudo apt install ffmpeg
```

---
## 🚀 Usage

### #️⃣ Download the Demo Video
```bash
    python video_download.py
```
### 2️⃣ Run the Traffic Analysis Notebook
Open JupyterLab or Jupyter Notebook:

```bash
    jupyter lab
```
Run `TrafficAnalysis.ipynb` step-by-step.

## 📊 Output
- **Live annotated video feed with:**
  - Lane dividers
  - Vehicle bounding boxes
  - Vehicle tracker IDs
  - Lane-wise vehicle counts

- **CSV log containing:**
  - Frame number
  - Timestamp
  - Lane-wise counts
  - Total vehicle count

---
## 🛠 Customization
- Modify `LANE_X` in `TrafficAnalysis.ipynb` to set vertical lane dividers.
- Change `COUNT_LINE_Y` to adjust the counting line position.

---
## 📌 Requirements
- Python 3.10+

- Ultralytics YOLO

- Supervision

- OpenCV

- NumPy

- yt-dlp

- FFmpeg (for merging downloaded streams)

---
## 📜 License
This project is open-source under the MIT License.

---
## ✨ Acknowledgments
- Ultralytics YOLO

- Supervision

- ByteTrack

- yt-dlp

---
## 🧑‍💻 Author
**Kashi Nath Chourasia**  
📧 [kashi533864@gmail.com](mailto:kashi533864@gmail.com)  
[🔗 LinkedIn](https://www.linkedin.com/in/kashi-nath-chourasia-42a39525a)
