import cv2
import mediapipe as mp
import numpy as np
import pickle
import os

# --- Configuration ---
SEGMENTS_FOLDER = "bharatanatyam_segments" # The folder where split videos are
OUTPUT_KEYPOINTS_FILE = "segmented_dance_keypoints.pkl" # Master file for all segment keypoints
DISPLAY_VIDEO_DURING_EXTRACTION = False # Set to True if you want to see extraction for each segment

# --- Initialize MediaPipe Pose ---
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils 

# --- Function to extract pose landmarks from a frame ---
def extract_landmarks_from_frame(frame, pose_model):
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose_model.process(image_rgb)
    
    landmarks = None
    if results.pose_landmarks:
        landmarks = np.array([[lm.x, lm.y, lm.z] for lm in results.pose_landmarks.landmark])
    
    return landmarks, results

def process_single_video_for_keypoints(video_path, pose_model, display_video):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  Error: Could not open video file '{video_path}'")
        return None

    video_keypoints_sequence = []
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        landmarks, results = extract_landmarks_from_frame(frame, pose_model)
        video_keypoints_sequence.append(landmarks) 
        
        if display_video:
            if results and results.pose_landmarks: 
                mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            cv2.imshow('Extracting Segment', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'): # Allow early quit for display
                break

    cap.release()
    if display_video:
        cv2.destroyWindow('Extracting Segment')
    
    return video_keypoints_sequence

# --- Main script execution ---
if __name__ == "__main__":
    if not os.path.exists(SEGMENTS_FOLDER):
        print(f"Error: Segments folder '{SEGMENTS_FOLDER}' not found. Please run split_video.py first.")
        exit()

    all_segments_keypoints = {} # Dictionary to store keypoints for all segments

    # Get sorted list of video files in the segments folder
    video_files = sorted([f for f in os.listdir(SEGMENTS_FOLDER) if f.endswith(('.mp4', '.avi', '.mov'))])

    if not video_files:
        print(f"No video files found in '{SEGMENTS_FOLDER}'.")
        exit()

    print(f"Starting keypoint extraction for videos in '{SEGMENTS_FOLDER}'...")

    for video_filename in video_files:
        video_path = os.path.join(SEGMENTS_FOLDER, video_filename)
        print(f"  Processing: {video_filename}")
        
        keypoints = process_single_video_for_keypoints(video_path, pose, DISPLAY_VIDEO_DURING_EXTRACTION)
        
        if keypoints is not None:
            all_segments_keypoints[video_filename] = keypoints
            print(f"  Extracted {len(keypoints)} frames for '{video_filename}'.")
        else:
            print(f"  Failed to extract keypoints or no pose detected in '{video_filename}'.")

    if all_segments_keypoints:
        with open(OUTPUT_KEYPOINTS_FILE, "wb") as f:
            pickle.dump(all_segments_keypoints, f)
        print(f"\nSuccessfully extracted and saved keypoints for {len(all_segments_keypoints)} segments to '{OUTPUT_KEYPOINTS_FILE}'.")
    else:
        print("\nNo keypoints extracted for any segment. Check videos and MediaPipe detection.")