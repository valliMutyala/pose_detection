

# Bharatanatyam Real-Time Pose Tutor

This project provides an interactive real-time dance tutor application that uses computer vision and machine learning (MediaPipe Pose) to compare a user's live dance movements to a pre-recorded Bharatanatyam dance sequence. The application guides the user segment by segment, providing live similarity feedback and progressing to the next part only upon achieving a defined mastery threshold for the current segment.

## Features

  * **Real-time Pose Detection:** Utilizes MediaPipe Pose to accurately detect 33 3D body landmarks.
  * **Segment-based Learning:** Breaks down a long dance video into manageable segments for focused practice.
  * **Live Comparison:** Compares the user's pose from a webcam feed against the pre-recorded reference dance.
  * **Similarity Feedback:** Provides a real-time cosine similarity score to indicate how closely the user matches the reference.
  * **Interactive Progression:** Automatically advances to the next dance segment only when the user achieves a defined mastery percentage for the current segment.
  * **Looping Practice:** If a segment is not mastered, it loops repeatedly, allowing for continuous practice.
  * **Audio Synchronization:** Plays the original dance video's audio synchronized with its video, providing a complete immersive experience.
  * **Video Splitting Utility:** Includes a script to easily split long dance videos into shorter segments.
  * **YouTube Downloader (Optional):** A utility script to download YouTube videos for use as reference material.

## Prerequisites

Before you begin, ensure you have the following installed on your Windows system:

1.  **Python 3.8 - 3.11:** Python 3.13 is very new and can have compatibility issues with some libraries. Python 3.8-3.11 are generally more stable for computer vision and ML libraries. If you are using Python 3.13, you might need to troubleshoot more.

      * Download from: [python.org](https://www.python.org/downloads/)
      * **Crucial:** During Python installation, make sure to check the box that says "Add Python to PATH".

2.  **Webcam:** A functional webcam connected to your computer.

3.  **FFmpeg:** This is an essential external tool that MoviePy (for splitting videos) and FFpyplayer (for audio playback) rely on.

      * **Download:** Go to [FFmpeg Downloads](https://ffmpeg.org/download.html). For Windows, click on the Windows icon and download a pre-built executable from `gyan.dev` (choose a "release full" build, usually a `.7z` file) or `BtbN/FFmpeg-Builds` (choose a `win64-gpl-static.zip` build).
      * **Extract:** Unzip the downloaded file. You'll get a folder (e.g., `ffmpeg-release-full`).
      * **Move to Stable Location:** Move this extracted folder (you can rename it simply to `ffmpeg`) to a stable, permanent location, for example: `C:\ffmpeg`.
      * **Add to System PATH (CRUCIAL):**
        1.  Copy the full path to the `bin` folder inside your `ffmpeg` directory (e.g., `C:\ffmpeg\bin`).
        2.  Search for "Environment Variables" in your Windows search bar and select "Edit the system environment variables."
        3.  Click the "Environment Variables..." button.
        4.  Under "System variables" (the bottom list), find `Path`, select it, and click "Edit...".
        5.  Click "New" and paste the path to your FFmpeg `bin` folder.
        6.  Click "OK" on all open windows to save.
        7.  **Restart your terminal (VS Code terminal, Command Prompt, PowerShell) completely** for changes to take effect.
      * **Verify Installation:** Open a **new** terminal and type `ffmpeg -version`. If it shows version info, you're good.

## Setup Instructions

Follow these steps to set up and run the project:

1.  **Clone the Repository:**

    ```bash
    git clone https://github.com/your-username/your-repo-name.git # Replace with your actual repo URL
    cd your-repo-name # Navigate into the project directory
    ```

    (If you are not using Git, simply download the project ZIP and extract it to your desired folder, then navigate into it via Command Prompt/PowerShell.)

2.  **Create a Python Virtual Environment:**
    It's highly recommended to use a virtual environment to manage project dependencies.

    ```bash
    python -m venv .venv
    ```

3.  **Activate the Virtual Environment:**

      * **On Windows (PowerShell/Command Prompt):**
        ```bash
        .\.venv\Scripts\Activate.ps1
        ```
      * You should see `(.venv)` at the beginning of your terminal prompt, indicating the environment is active.

4.  **Install Required Python Libraries:**
    First, create a `requirements.txt` file in your project's root directory and add the following content:

    ```
    opencv-python
    mediapipe
    numpy
    moviepy==1.0.3  # Crucial: Pin to 1.x version for 'moviepy.editor'
    ffpyplayer
    yt-dlp
    pillow
    imageio
    imageio_ffmpeg
    decorator
    proglog
    python-dotenv
    tqdm
    colorama
    ```

    Then, install them using pip:

    ```bash
    pip install -r requirements.txt
    ```

## Project Workflow & Usage

The project runs in a sequential workflow. Ensure your virtual environment is activated for all commands.

### Step 1: Prepare Your Reference Dance Video

Place your full, long Bharatanatyam dance video (e.g., `bharatanatyam.mp4`) in the root directory of your project.

*(Optional: If your video is on YouTube, you can use the provided downloader script)*

```bash
python download_youtube_video.py
```

Follow the prompts to enter the YouTube URL. The video will be saved as `my_reference_dance.mp4` by default. Rename it to `bharatanatyam.mp4` or update the script configurations accordingly.

### Step 2: Split the Video into Segments

This script will take your long `bharatanatyam.mp4` and split it into 5-second video chunks.

1.  Open `split_video.py`.
2.  Verify/update the `input_video` variable to match your video's filename (e.g., `"bharatanatyam.mp4"`).
3.  Run the script:
    ```bash
    python split_video.py
    ```
    This will create a new folder named `bharatanatyam_segments` containing files like `bharatanatyam_part001.mp4`, `bharatanatyam_part002.mp4`, etc.

### Step 3: Extract Keypoints from Dance Segments

This script processes each 5-second video segment and saves all detected pose keypoints into a single `.pkl` file.

1.  Open `extract_segmented_dance_keypoints.py`.
2.  Verify that `SEGMENTS_FOLDER` is correctly set to `"bharatanatyam_segments"`.
3.  Run the script:
    ```bash
    python extract_segmented_dance_keypoints.py
    ```
    This will create `segmented_dance_keypoints.pkl` in your project's root directory.

### Step 4: Run the Live Dance Comparison

This is the main application. It uses your webcam, plays the segmented dance, and provides live feedback.

  Run the application:

    ```bash
    python live_segmented_dance_comparison.py
    ```

      * **Expect two windows to appear:** "Your Live Performance" (webcam) and "Reference Dance Segment" (the dance video).
      * The dance will play segment by segment. Try to imitate the movements.
      * The application will provide real-time similarity scores and feedback.
      * It will loop the current segment if your mastery is below the `SEGMENT_MASTERY_THRESHOLD_PERCENT`.
      * Press `q` on either window to quit the application.
```

## Troubleshooting

  * **`ModuleNotFoundError`:**
      * Ensure your virtual environment is active: `(.venv)` should be in your terminal prompt.
      * Run `pip install -r requirements.txt` again to confirm all libraries are installed.
      * If specifically for `moviepy.editor`, ensure `moviepy==1.0.3` is installed (check `pip list`).
  * **`ffmpeg not recognized`:**
      * FFmpeg is not correctly installed or its `bin` directory is not in your system's PATH. Revisit **Prerequisites Step 3**. Remember to **restart your terminal** after adding to PATH.
  * **Blank Video Window / `Assertion failed size.width>0` error:**
      * The video file might be corrupted, or OpenCV struggles with its codec. Ensure the video (`.mp4` segments) plays fine in VLC media player.
      * Verify the `REFERENCE_VIDEO_PATH` (in `extract_segmented_dance_keypoints.py`) and `SEGMENTS_FOLDER` (in `live_segmented_dance_comparison.py`) are absolutely correct.
  * **Audio Desynchronization / Looping Audio:**
      * This is the most common and challenging issue. **The primary solution is to tune `TARGET_APP_FPS` in `live_segmented_dance_comparison.py` downwards.** Your system might not be powerful enough to run MediaPipe and all other processes at a higher FPS. Try `15`, `10`, or `7`.
      * Consider implementing the optional `cv2.resize()` optimization in `extract_landmarks_from_frame` to reduce input resolution to MediaPipe.
      * Ensure your video file has proper audio (AAC codec is recommended for MP4 segments).
  * **Webcam Not Found / `Could not open webcam`:**
      * Ensure no other application is using your webcam.
      * Check your webcam drivers.
      * If you have multiple webcams, you can try changing `cv2.VideoCapture(0)` to `cv2.VideoCapture(1)`, `cv2.VideoCapture(2)`, etc.
