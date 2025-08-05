import cv2
import mediapipe as mp
import numpy as np
import pickle
import time
import os

# --- Configuration (GLOBAL CONSTANTS) ---
DISPLAY_WINDOW_WIDTH = 800
DISPLAY_WINDOW_HEIGHT = 600
SIMILARITY_THRESHOLD = 0.60
WINDOW_NAME = "Bharatanatyam Practice" # ⭐ Use a single, consistent window name

INSTRUCTION_MESSAGE = "Match the dancer's pose!"
SUCCESS_MESSAGE = "Excellent!"
NO_POSE_MESSAGE = "Pose not detected in you or reference."
NO_USER_POSE_MESSAGE = "Your pose not detected!"
NO_REF_POSE_MESSAGE = "Reference pose not detected in this frame!"
COMPLETED_ALL_SEGMENTS_MESSAGE = "Congratulations! You mastered the entire dance!"
REFERENCE_KEYPOINTS_FILE = "segmented_dance_keypoints.pkl"
SEGMENTS_FOLDER = "bharatanatyam_segments"

# --- Setup and Loading ---
try:
    with open(REFERENCE_KEYPOINTS_FILE, "rb") as f:
        all_segments_keypoints = pickle.load(f)
except FileNotFoundError:
    print(f"❌ Error: '{REFERENCE_KEYPOINTS_FILE}' not found. Run keypoint extraction script first.")
    exit()

segment_filenames = sorted(all_segments_keypoints.keys())
if not segment_filenames:
    print("❌ Error: No keypoints found in the loaded file. Exiting.")
    exit()

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# (Helper functions like extract_landmarks, calculate_angle, etc. remain the same)
def extract_landmarks_from_frame(frame, pose_model):
    """Extracts pose landmarks from a given frame."""
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose_model.process(image_rgb)
    if results.pose_landmarks:
        return np.array([[lm.x, lm.y, lm.z] for lm in results.pose_landmarks.landmark]), results
    return None, results

def calculate_angle(p1, p2, p3):
    """Calculates the angle between three points using only X and Y coordinates for stability."""
    p1 = p1[:2]
    p2 = p2[:2]
    p3 = p3[:2]
    
    v1 = p1 - p2
    v2 = p3 - p2
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    if norm_v1 == 0 or norm_v2 == 0: return 0.0
    angle_rad = np.arccos(np.clip(dot_product / (norm_v1 * norm_v2), -1.0, 1.0))
    return np.degrees(angle_rad)

def get_pose_angles(keypoints):
    """Extracts key joint angles from the pose landmarks."""
    if keypoints is None or len(keypoints) < 33: return None
    angles = {
        'right_elbow': calculate_angle(keypoints[12], keypoints[14], keypoints[16]),
        'left_elbow': calculate_angle(keypoints[11], keypoints[13], keypoints[15]),
        'right_shoulder': calculate_angle(keypoints[24], keypoints[12], keypoints[14]),
        'left_shoulder': calculate_angle(keypoints[23], keypoints[11], keypoints[13]),
        'right_knee': calculate_angle(keypoints[24], keypoints[26], keypoints[28]),
        'left_knee': calculate_angle(keypoints[23], keypoints[25], keypoints[27]),
        'right_hip': calculate_angle(keypoints[12], keypoints[24], keypoints[26]),
        'left_hip': calculate_angle(keypoints[11], keypoints[23], keypoints[25]),
    }
    return angles

def compare_poses_by_angles(ref_keypoints, live_keypoints):
    """Compares poses using a Gaussian function for a more natural similarity score."""
    ref_angles = get_pose_angles(ref_keypoints)
    live_angles = get_pose_angles(live_keypoints)
    if ref_angles is None or live_angles is None: return 0.0
    tolerance_degrees = 25.0
    total_similarity = 0.0
    for joint, ref_angle in ref_angles.items():
        live_angle = live_angles.get(joint)
        if live_angle is not None:
            angle_diff = abs(ref_angle - live_angle)
            similarity = np.exp(- (angle_diff**2) / (2 * tolerance_degrees**2))
            total_similarity += similarity
    return total_similarity / len(ref_angles) if ref_angles else 0.0

def check_segment_mastery(similarity_scores, threshold):
    """Checks if the average similarity score for the segment meets the required threshold."""
    if not similarity_scores: return False
    avg_score = np.mean(similarity_scores)
    print(f"Required Similarity: {threshold*100:.0f}%, Average Score: {avg_score*100:.2f}%")
    if avg_score >= threshold:
        print("✅ Segment passed. Proceeding to next segment.")
        return True
    else:
        print("❌ Segment failed. Repeating segment.")
        return False

def display_countdown(cap, duration=3):
    """Displays a countdown in the main window."""
    start_time = time.time()
    while time.time() - start_time < duration:
        ret, frame = cap.read()
        if not ret: break
        
        frame = cv2.flip(frame, 1)
        # We only need a small window for the countdown message
        display_frame = np.zeros((DISPLAY_WINDOW_HEIGHT, DISPLAY_WINDOW_WIDTH * 2, 3), dtype=np.uint8)
        
        remaining_time = duration - int(time.time() - start_time)
        text = f"Get ready... {remaining_time}"
        
        # ⭐ Center the text properly in the combined window
        (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_TRIPLEX, 2, 3)
        text_x = (display_frame.shape[1] - text_width) // 2
        text_y = (display_frame.shape[0] + text_height) // 2
        
        cv2.putText(display_frame, text, (text_x, text_y), cv2.FONT_HERSHEY_TRIPLEX, 2, (0, 255, 255), 3)
        cv2.imshow(WINDOW_NAME, display_frame) # ⭐ Use the consistent window name
        
        if cv2.waitKey(1) & 0xFF == ord('q'): return False
    return True

# --- Main Application Loop ---
cap_live = cv2.VideoCapture(0)
if not cap_live.isOpened():
    print("❌ Error: Could not open webcam.")
    exit()

# ⭐ Create the window once before the loop starts
cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL) 

current_segment_index = 0
while current_segment_index < len(segment_filenames):
    segment_filename = segment_filenames[current_segment_index]
    segment_video_path = os.path.join(SEGMENTS_FOLDER, segment_filename)
    current_segment_keypoints = all_segments_keypoints[segment_filename]

    cap_ref_segment = cv2.VideoCapture(segment_video_path)
    if not cap_ref_segment.isOpened():
        print(f"Error opening segment video '{segment_video_path}'. Skipping.")
        current_segment_index += 1
        continue
        
    # ⭐ --- Speed Synchronization --- ⭐
    # Get the FPS of the reference video to control playback speed
    REF_VIDEO_FPS = cap_ref_segment.get(cv2.CAP_PROP_FPS) or 30.0
    FRAME_DELAY_MS = int(1000 / REF_VIDEO_FPS)
    
    print(f"\n--- Starting Segment {current_segment_index + 1}/{len(segment_filenames)}: '{segment_filename}' at {REF_VIDEO_FPS:.1f} FPS ---")
    
    segment_passed = False
    while not segment_passed:
        if not display_countdown(cap_live, duration=3):
            exit()

        current_segment_frame_index = 0
        similarity_scores = []
        cap_ref_segment.set(cv2.CAP_PROP_POS_FRAMES, 0)

        while cap_ref_segment.isOpened():
            ret_ref, ref_frame = cap_ref_segment.read()
            ret_live, live_frame = cap_live.read()
            
            if not ret_ref or not ret_live:
                break

            live_frame_flipped = cv2.flip(live_frame, 1)

            display_live_frame = cv2.resize(live_frame_flipped, (DISPLAY_WINDOW_WIDTH, DISPLAY_WINDOW_HEIGHT))
            display_ref_frame = cv2.resize(ref_frame, (DISPLAY_WINDOW_WIDTH, DISPLAY_WINDOW_HEIGHT))

            live_keypoints, live_pose_results = extract_landmarks_from_frame(live_frame_flipped, pose)
            ref_keypoints = current_segment_keypoints[current_segment_frame_index] if current_segment_frame_index < len(current_segment_keypoints) else None

            sim = compare_poses_by_angles(ref_keypoints, live_keypoints)
            if live_keypoints is not None and ref_keypoints is not None:
                similarity_scores.append(sim)

            # --- UI Overlays (Logic remains the same) ---
            progress = (current_segment_frame_index + 1) / len(current_segment_keypoints)
            cv2.rectangle(display_live_frame, (0, 0), (int(DISPLAY_WINDOW_WIDTH * progress), 10), (0, 255, 0), -1)
            # Message logic... (omitted for brevity, it's correct)
            if live_keypoints is not None and ref_keypoints is not None:
                message_color = (0, 255, 0) if sim >= SIMILARITY_THRESHOLD else (0, 165, 255)
                feedback_message = SUCCESS_MESSAGE if sim >= SIMILARITY_THRESHOLD else INSTRUCTION_MESSAGE
            elif live_keypoints is None:
                message_color = (0, 0, 255)
                feedback_message = NO_USER_POSE_MESSAGE
            else:
                message_color = (0, 0, 255)
                feedback_message = NO_REF_POSE_MESSAGE
            
            cv2.putText(display_live_frame, f'Similarity: {sim:.1%}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, message_color, 2)
            cv2.putText(display_live_frame, feedback_message, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, message_color, 2)
            if live_pose_results and live_pose_results.pose_landmarks:
                mp_drawing.draw_landmarks(display_live_frame, live_pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            
            combined_frame = np.hstack((display_ref_frame, display_live_frame))
            cv2.imshow(WINDOW_NAME, combined_frame) # ⭐ Use the consistent window name

            current_segment_frame_index += 1
            
            # ⭐ Use the calculated frame delay to ensure correct speed
            if cv2.waitKey(FRAME_DELAY_MS) & 0xFF == ord('q'):
                exit()
        
        segment_passed = check_segment_mastery(similarity_scores, SIMILARITY_THRESHOLD)

    cap_ref_segment.release()
    if segment_passed:
        current_segment_index += 1

# --- Final Message ---
cap_live.release()
cv2.destroyAllWindows()
print(COMPLETED_ALL_SEGMENTS_MESSAGE)