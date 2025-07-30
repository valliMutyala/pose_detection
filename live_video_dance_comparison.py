import cv2
import mediapipe as mp
import numpy as np
import pickle
import time
import os
from ffpyplayer.player import MediaPlayer 

# --- Configuration (GLOBAL CONSTANTS) ---
DISPLAY_WINDOW_WIDTH = 640
SIMILARITY_THRESHOLDS = [0.70, 0.85, 0.95]  # Stage thresholds
INSTRUCTION_MESSAGE = "Match the dancer's pose!"
SUCCESS_MESSAGE = "Great job! Hold or prepare for next part!"
NO_POSE_MESSAGE = "Pose not detected in you or reference."
NO_USER_POSE_MESSAGE = "Your pose not detected!"
NO_REF_POSE_MESSAGE = "Reference pose not detected in this frame!"
COMPLETED_ALL_SEGMENTS_MESSAGE = "Congratulations! You mastered the entire dance!"
REFERENCE_KEYPOINTS_FILE = "segmented_dance_keypoints.pkl"
SEGMENTS_FOLDER = "bharatanatyam_segments"

try:
    with open(REFERENCE_KEYPOINTS_FILE, "rb") as f:
        all_segments_keypoints = pickle.load(f)
except FileNotFoundError:
    print(f"Error: '{REFERENCE_KEYPOINTS_FILE}' not found. Run keypoint extraction script first.")
    exit()

segment_filenames = sorted(all_segments_keypoints.keys())
if not segment_filenames:
    print("Error: No keypoints found in the loaded file. Exiting.")
    exit()

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils 

def extract_landmarks_from_frame(frame, pose_model):
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose_model.process(image_rgb)
    if results.pose_landmarks:
        return np.array([[lm.x, lm.y, lm.z] for lm in results.pose_landmarks.landmark]), results
    return None, results

def cosine_similarity(a, b):
    a, b = a.flatten(), b.flatten()
    norm_a, norm_b = np.linalg.norm(a), np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return np.dot(a, b) / (norm_a * norm_b)

def staged_mastery_check(similarity_scores, thresholds):
    avg_score = np.mean(similarity_scores)
    stage_1_thresh, stage_2_thresh, stage_3_thresh = thresholds[0], thresholds[1], 0.92  # Loosen Stage 3

    print(f"Stage 1: Required ≥ {stage_1_thresh*100:.0f}%, Got = {avg_score*100:.2f}%")
    if avg_score < stage_1_thresh:
        print("❌ Failed Stage 1. Repeating segment.")
        return False

    print(f"Stage 2: Required ≥ {stage_2_thresh*100:.0f}%, Got = {avg_score*100:.2f}%")
    if avg_score < stage_2_thresh:
        print("❌ Failed Stage 2. Repeating segment.")
        return False

    print(f"Stage 3: Required ≥ {stage_3_thresh*100:.0f}%, Got = {avg_score*100:.2f}%")
    if avg_score < stage_3_thresh:
        print("❌ Failed Stage 3. Repeating segment.")
        return False

    print("✅ All stages passed. Proceeding to next segment.")
    return True


cap_live = cv2.VideoCapture(0)
if not cap_live.isOpened():
    print("Error: Could not open webcam.")
    exit()

live_width = int(cap_live.get(cv2.CAP_PROP_FRAME_WIDTH))
live_height = int(cap_live.get(cv2.CAP_PROP_FRAME_HEIGHT))
live_aspect_ratio = live_width / live_height
calculated_live_display_height = int(DISPLAY_WINDOW_WIDTH / live_aspect_ratio)

current_segment_index = 0
while current_segment_index < len(segment_filenames):
    segment_filename = segment_filenames[current_segment_index]
    segment_video_path = os.path.join(SEGMENTS_FOLDER, segment_filename)
    current_segment_keypoints = all_segments_keypoints[segment_filename]

    cap_ref_segment = cv2.VideoCapture(segment_video_path)
    if not cap_ref_segment.isOpened():
        print(f"Error: Could not open segment video '{segment_video_path}'. Skipping.")
        current_segment_index += 1
        continue

    player_segment = MediaPlayer(segment_video_path)

    ref_width = int(cap_ref_segment.get(cv2.CAP_PROP_FRAME_WIDTH))
    ref_height = int(cap_ref_segment.get(cv2.CAP_PROP_FRAME_HEIGHT))
    ref_aspect_ratio = ref_width / ref_height
    calculated_ref_display_height = int(DISPLAY_WINDOW_WIDTH / ref_aspect_ratio)

    REF_VIDEO_FPS = cap_ref_segment.get(cv2.CAP_PROP_FPS) or 30.0
    FRAME_DELAY_MS = 1000 / REF_VIDEO_FPS

    print(f"\n--- Starting Segment {current_segment_index + 1}/{len(segment_filenames)}: '{segment_filename}' ---")

    segment_passed = False
    while not segment_passed:
        current_segment_frame_index = 0
        similarity_scores = []

        cap_ref_segment.set(cv2.CAP_PROP_POS_FRAMES, 0)
        player_segment.close_player()
        player_segment = MediaPlayer(segment_video_path)

        while cap_ref_segment.isOpened():
            ret_ref, ref_frame = cap_ref_segment.read()
            ret_live, live_frame = cap_live.read()
            audio_frame, val = player_segment.get_frame()

            if not ret_ref or not ret_live:
                break

            live_frame = cv2.flip(live_frame, 1)

            display_live_frame = cv2.resize(live_frame, (DISPLAY_WINDOW_WIDTH, calculated_live_display_height))
            display_ref_frame = cv2.resize(ref_frame, (DISPLAY_WINDOW_WIDTH, calculated_ref_display_height))

            live_keypoints, live_pose_landmarks_obj = extract_landmarks_from_frame(live_frame, pose)
            ref_keypoints = current_segment_keypoints[current_segment_frame_index] if current_segment_frame_index < len(current_segment_keypoints) else None

            sim = 0.0
            feedback_message = NO_POSE_MESSAGE
            message_color = (0, 0, 255)

            if live_keypoints is not None and ref_keypoints is not None:
                sim = cosine_similarity(ref_keypoints, live_keypoints)
                similarity_scores.append(sim)
                if sim >= SIMILARITY_THRESHOLDS[0]:
                    feedback_message = SUCCESS_MESSAGE
                    message_color = (0, 255, 0)
                else:
                    feedback_message = INSTRUCTION_MESSAGE
                    message_color = (0, 165, 255)
            elif live_keypoints is None and ref_keypoints is not None:
                feedback_message = NO_USER_POSE_MESSAGE
            elif live_keypoints is not None and ref_keypoints is None:
                feedback_message = NO_REF_POSE_MESSAGE

            cv2.putText(display_live_frame, f'Similarity: {sim:.2f}', (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, message_color, 2)
            cv2.putText(display_live_frame, feedback_message, (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, message_color, 2)
            if live_pose_landmarks_obj and live_pose_landmarks_obj.pose_landmarks:
                mp_drawing.draw_landmarks(display_live_frame, live_pose_landmarks_obj.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            cv2.imshow("Your Live Performance", display_live_frame)
            cv2.imshow("Reference Dance Segment", display_ref_frame)

            current_segment_frame_index += 1

            if current_segment_frame_index >= len(current_segment_keypoints):
                segment_passed = staged_mastery_check(similarity_scores, SIMILARITY_THRESHOLDS)
                break

            if cv2.waitKey(1) & 0xFF == ord('q'):
                cap_live.release()
                cap_ref_segment.release()
                player_segment.close_player()
                cv2.destroyAllWindows()
                exit()

    cap_ref_segment.release()
    player_segment.close_player()
    current_segment_index += 1

final_msg = np.zeros((calculated_live_display_height, DISPLAY_WINDOW_WIDTH, 3), dtype=np.uint8)
cv2.putText(final_msg, COMPLETED_ALL_SEGMENTS_MESSAGE, (10, calculated_live_display_height // 2),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
cv2.imshow("Final Message", final_msg)
cv2.waitKey(0)
cap_live.release()
cv2.destroyAllWindows()
