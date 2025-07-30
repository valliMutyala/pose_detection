import cv2
import mediapipe as mp
import numpy as np
import pickle
import os

# --- Configuration ---
SEGMENTS_FOLDER = "bharatanatyam_segments"  # Folder containing split video segments
OUTPUT_KEYPOINTS_FILE = "segmented_dance_keypoints.pkl"  # File to store extracted keypoints
DISPLAY_VIDEO_DURING_EXTRACTION = False  # Set True to visually debug the pose extraction

# --- Initialize MediaPipe Pose ---
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False,
                    min_detection_confidence=0.6,
                    min_tracking_confidence=0.6)
mp_drawing = mp.solutions.drawing_utils


def extract_landmarks_from_frame(frame, pose_model):
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose_model.process(image_rgb)

    if results.pose_landmarks:
        landmarks = np.array([[lm.x, lm.y, lm.z] for lm in results.pose_landmarks.landmark])
        return landmarks, results
    else:
        return None, results


def process_single_video_for_keypoints(video_path, pose_model, display_video=False):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  ❌ Could not open: {video_path}")
        return None

    keypoints_sequence = []
    frame_count = 0
    valid_frames = 0

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
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    if display_video:
        cv2.destroyWindow("Pose Extraction")

    if valid_frames == 0:
        return None

    print(f"    ➜ {valid_frames}/{frame_count} valid frames extracted.")
    return keypoints_sequence


def main():
    if not os.path.exists(SEGMENTS_FOLDER):
        print(f"❗ Folder not found: {SEGMENTS_FOLDER}. Run split_video.py first.")
        return

    video_files = sorted([f for f in os.listdir(SEGMENTS_FOLDER)
                          if f.lower().endswith(('.mp4', '.avi', '.mov'))])

    if not video_files:
        print(f"❗ No video files found in: {SEGMENTS_FOLDER}")
        return

    print(f"🔍 Starting keypoint extraction from {len(video_files)} video(s)...\n")
    all_keypoints = {}

    for filename in video_files:
        path = os.path.join(SEGMENTS_FOLDER, filename)
        print(f"📹 Processing: {filename}")

        keypoints = process_single_video_for_keypoints(path, pose, DISPLAY_VIDEO_DURING_EXTRACTION)
        if keypoints:
            all_keypoints[filename] = keypoints
            print(f"    ✅ Saved {len(keypoints)} frames.")
        else:
            print(f"    ⚠️  No valid pose detected in: {filename}")

    if all_keypoints:
        with open(OUTPUT_KEYPOINTS_FILE, "wb") as f:
            pickle.dump(all_keypoints, f)
        print(f"\n✅ Keypoints saved to '{OUTPUT_KEYPOINTS_FILE}' for {len(all_keypoints)} video(s).")
    else:
        print("\n❌ No valid keypoints extracted from any video.")

if __name__ == "__main__":
    main()
