import cv2
import mediapipe as mp
import numpy as np
import pickle
import time
import os

# --- Configuration (GLOBAL CONSTANTS) ---
DISPLAY_WINDOW_WIDTH = 720
DISPLAY_WINDOW_HEIGHT = 560
SIMILARITY_THRESHOLD = 0.70# Adjusted for angle-based comparison
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
    """
    Extracts pose landmarks from a given frame using the MediaPipe Pose model.
    This version returns all three coordinates (x, y, z) to ensure compatibility
    with keypoints stored in the .pkl file.
    """
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose_model.process(image_rgb)
    if results.pose_landmarks:
        # Returns a numpy array of [x, y, z] coordinates for each landmark
        return np.array([[lm.x, lm.y, lm.z] for lm in results.pose_landmarks.landmark]), results
    return None, results

def calculate_angle(p1, p2, p3):
    """Calculates the angle between three points."""
    v1 = p1 - p2
    v2 = p3 - p2
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    if norm_v1 == 0 or norm_v2 == 0:
        return 0.0
    angle_rad = np.arccos(np.clip(dot_product / (norm_v1 * norm_v2), -1.0, 1.0))
    return np.degrees(angle_rad)

def get_pose_angles(keypoints):
    """Extracts key joint angles from the pose landmarks."""
    if keypoints is None or len(keypoints) < 33:
        return None
    
    angles = {}
    
    # Right arm
    angles['right_elbow'] = calculate_angle(keypoints[mp_pose.PoseLandmark.RIGHT_SHOULDER.value], 
                                            keypoints[mp_pose.PoseLandmark.RIGHT_ELBOW.value],
                                            keypoints[mp_pose.PoseLandmark.RIGHT_WRIST.value])
    angles['right_shoulder'] = calculate_angle(keypoints[mp_pose.PoseLandmark.RIGHT_HIP.value],
                                              keypoints[mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
                                              keypoints[mp_pose.PoseLandmark.RIGHT_ELBOW.value])
    
    # Left arm
    angles['left_elbow'] = calculate_angle(keypoints[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
                                           keypoints[mp_pose.PoseLandmark.LEFT_ELBOW.value],
                                           keypoints[mp_pose.PoseLandmark.LEFT_WRIST.value])
    angles['left_shoulder'] = calculate_angle(keypoints[mp_pose.PoseLandmark.LEFT_HIP.value],
                                             keypoints[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
                                             keypoints[mp_pose.PoseLandmark.LEFT_ELBOW.value])
    
    # Right leg
    angles['right_knee'] = calculate_angle(keypoints[mp_pose.PoseLandmark.RIGHT_HIP.value],
                                           keypoints[mp_pose.PoseLandmark.RIGHT_KNEE.value],
                                           keypoints[mp_pose.PoseLandmark.RIGHT_ANKLE.value])
    angles['right_hip'] = calculate_angle(keypoints[mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
                                          keypoints[mp_pose.PoseLandmark.RIGHT_HIP.value],
                                          keypoints[mp_pose.PoseLandmark.RIGHT_KNEE.value])
    
    # Left leg
    angles['left_knee'] = calculate_angle(keypoints[mp_pose.PoseLandmark.LEFT_HIP.value],
                                          keypoints[mp_pose.PoseLandmark.LEFT_KNEE.value],
                                          keypoints[mp_pose.PoseLandmark.LEFT_ANKLE.value])
    angles['left_hip'] = calculate_angle(keypoints[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
                                         keypoints[mp_pose.PoseLandmark.LEFT_HIP.value],
                                         keypoints[mp_pose.PoseLandmark.LEFT_KNEE.value])
    
    return angles

def compare_poses_by_angles(ref_keypoints, live_keypoints):
    """Compares poses by calculating the similarity of their key joint angles."""
    ref_angles = get_pose_angles(ref_keypoints)
    live_angles = get_pose_angles(live_keypoints)
    
    if ref_angles is None or live_angles is None:
        return 0.0

    score = 0.0
    num_joints = 0
    for joint in ref_angles:
        ref_angle = ref_angles[joint]
        live_angle = live_angles[joint]
        
        # Calculate the absolute difference between the angles
        angle_diff = abs(ref_angle - live_angle)
        
        # Normalize the difference to a 0-1 scale, where 1 is a perfect match
        # We assume a max reasonable angle difference is around 60 degrees for a good score
        similarity = 1.0 - (angle_diff / 60.0)
        score += max(0, similarity) # Cap at 0
        num_joints += 1

    if num_joints > 0:
        return score / num_joints
    return 0.0

def check_segment_mastery(similarity_scores, threshold):
    """
    Checks if the average similarity score for the segment meets the required threshold.
    """
    if not similarity_scores:
        print("Warning: No similarity scores to check.")
        return False
        
    avg_score = np.mean(similarity_scores)
    
    print(f"Required Similarity: {threshold*100:.0f}%, Average Score: {avg_score*100:.2f}%")
    if avg_score >= threshold:
        print("✅ Segment passed. Proceeding to next segment.")
        return True
    else:
        print("❌ Segment failed. Repeating segment.")
        return False

def display_countdown(cap, duration=3):
    """Displays a countdown on the live video window."""
    start_time = time.time()
    while time.time() - start_time < duration:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.flip(frame, 1)
        frame = cv2.resize(frame, (DISPLAY_WINDOW_WIDTH, DISPLAY_WINDOW_HEIGHT))
        
        remaining_time = duration - int(time.time() - start_time)
        text = f"Get ready... {remaining_time}"
        cv2.putText(frame, text, (int(DISPLAY_WINDOW_WIDTH/2) - 150, int(DISPLAY_WINDOW_HEIGHT/2)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)
        cv2.imshow("Your Live Performance", frame)
        
        # Keep the window responsive
        if cv2.waitKey(1) & 0xFF == ord('q'):
            return False
    
    # Final 'Go!' message
    ret, frame = cap.read()
    if ret:
        frame = cv2.flip(frame, 1)
        frame = cv2.resize(frame, (DISPLAY_WINDOW_WIDTH, DISPLAY_WINDOW_HEIGHT))
        cv2.putText(frame, "Go!", (int(DISPLAY_WINDOW_WIDTH/2) - 50, int(DISPLAY_WINDOW_HEIGHT/2)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
        cv2.imshow("Your Live Performance", frame)
        cv2.waitKey(500) # Show 'Go!' for half a second
    
    return True


# --- Main Application Loop ---
cap_live = cv2.VideoCapture(0)
if not cap_live.isOpened():
    print("Error: Could not open webcam.")
    exit()

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

    REF_VIDEO_FPS = cap_ref_segment.get(cv2.CAP_PROP_FPS) or 30.0
    FRAME_DELAY_MS = 1000 / REF_VIDEO_FPS

    print(f"\n--- Starting Segment {current_segment_index + 1}/{len(segment_filenames)}: '{segment_filename}' ---")
    

    segment_passed = False
    while not segment_passed:
        # Countdown break before each segment attempt
        if not display_countdown(cap_live, duration=3):
            segment_passed = True # Break outer loop if user presses 'q'
            break

        current_segment_frame_index = 0
        similarity_scores = []
        
        # Reset video for re-attempt
        cap_ref_segment.set(cv2.CAP_PROP_POS_FRAMES, 0)

        while cap_ref_segment.isOpened():
            ret_ref, ref_frame = cap_ref_segment.read()
            ret_live, live_frame = cap_live.read()
            
            if not ret_ref or not ret_live:
                break

            live_frame = cv2.flip(live_frame, 1)

            # Resize both frames to the new fixed dimensions
            display_live_frame = cv2.resize(live_frame, (DISPLAY_WINDOW_WIDTH, DISPLAY_WINDOW_HEIGHT))
            display_ref_frame = cv2.resize(ref_frame, (DISPLAY_WINDOW_WIDTH, DISPLAY_WINDOW_HEIGHT))

            live_keypoints, live_pose_landmarks_obj = extract_landmarks_from_frame(live_frame, pose)
            ref_keypoints = current_segment_keypoints[current_segment_frame_index] if current_segment_frame_index < len(current_segment_keypoints) else None

            sim = 0.0
            feedback_message = NO_POSE_MESSAGE
            message_color = (0, 0, 255)

            if live_keypoints is not None and ref_keypoints is not None:
                sim = compare_poses_by_angles(ref_keypoints, live_keypoints)
                similarity_scores.append(sim)
                if sim >= SIMILARITY_THRESHOLD:
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
                segment_passed = check_segment_mastery(similarity_scores, SIMILARITY_THRESHOLD)
                break

            if cv2.waitKey(1) & 0xFF == ord('q'):
                cap_live.release()
                cap_ref_segment.release()
                cv2.destroyAllWindows()
                exit()

    cap_ref_segment.release()
    current_segment_index += 1

final_msg = np.zeros((DISPLAY_WINDOW_HEIGHT, DISPLAY_WINDOW_WIDTH, 3), dtype=np.uint8)
cv2.putText(final_msg, COMPLETED_ALL_SEGMENTS_MESSAGE, (10, DISPLAY_WINDOW_HEIGHT // 2),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
cv2.imshow("Final Message", final_msg)
cv2.waitKey(0)
cap_live.release()
cv2.destroyAllWindows()
