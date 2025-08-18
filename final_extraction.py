import cv2
import mediapipe as mp
import numpy as np
import pickle
import os

# --- Configuration ---
REFERENCE_VIDEO_PATH = "bharatanatyam.mp4"  # IMPORTANT: Change this to your video file's name
OUTPUT_KEYPOINTS_FILE = "dance_keypoints.pkl"      # File to store the extracted keypoints
DISPLAY_VIDEO_DURING_EXTRACTION = True            # Set to True to see the extraction process

# --- Initialize MediaPipe Pose ---
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False,
                    min_detection_confidence=0.6,
                    min_tracking_confidence=0.6)
mp_drawing = mp.solutions.drawing_utils


def extract_landmarks_from_frame(frame, pose_model):
    """Extracts pose landmarks from a given frame."""
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose_model.process(image_rgb)
    if results.pose_landmarks:
        landmarks = np.array([[lm.x, lm.y, lm.z] for lm in results.pose_landmarks.landmark])
        return landmarks, results
    return None, results


def process_video_for_keypoints(video_path, pose_model, display_video=False):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Could not open video: {video_path}")
        return None

    keypoints_sequence = []
    frame_count = 0
    valid_frames = 0
    ref_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_delay_ms = int(1000 / ref_fps)

    print(f"🔍 Starting keypoint extraction for '{video_path}' at {ref_fps:.2f} FPS.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        landmarks, results = extract_landmarks_from_frame(frame, pose_model)

        if landmarks is not None:
            keypoints_sequence.append(landmarks)
            valid_frames += 1

        if display_video:
            if results and results.pose_landmarks:
                mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            cv2.imshow("Pose Extraction", frame)
            
            # Use the video's FPS to control playback speed
            if cv2.waitKey(frame_delay_ms) & 0xFF == ord('q'):
                break

    cap.release()
    if display_video:
        cv2.destroyWindow("Pose Extraction")

    if valid_frames == 0:
        return None

    print(f"✅ Extracted {valid_frames}/{frame_count} valid frames.")
    return keypoints_sequence


def main():
    if not os.path.exists(REFERENCE_VIDEO_PATH):
        print(f"❗ Error: The video file '{REFERENCE_VIDEO_PATH}' was not found.")
        print("Please ensure your video file is in the same folder as this script and that the file name is correct.")
        return

    keypoints = process_video_for_keypoints(REFERENCE_VIDEO_PATH, pose, DISPLAY_VIDEO_DURING_EXTRACTION)
    
    if keypoints:
        with open(OUTPUT_KEYPOINTS_FILE, "wb") as f:
            pickle.dump(keypoints, f)
        print(f"\n✅ Keypoints successfully saved to '{OUTPUT_KEYPOINTS_FILE}'.")
    else:
        print("\n❌ No valid keypoints could be extracted from the video.")

if __name__ == "__main__":
    main()