import cv2
import mediapipe as mp
import numpy as np
import pickle
import time

# --- Configuration ---
REFERENCE_VIDEO_PATH = "bharatanatyam.mp4"
REFERENCE_KEYPOINTS_FILE = "dance_keypoints.pkl"
DISPLAY_WINDOW_WIDTH = 720
DISPLAY_WINDOW_HEIGHT = 560
SIMILARITY_THRESHOLD = 0.7
POSE_WAIT_SECONDS = 5
FRAME_WINDOW = 5  # Number of frames before and after for tolerance

INSTRUCTION_MESSAGE = "Get Ready! The dance will start in 5 seconds."
SUCCESS_MESSAGE = "Great job!"
FAIL_MESSAGE = "Try matching the pose again!"

# --- Load reference keypoints ---
with open(REFERENCE_KEYPOINTS_FILE, "rb") as f:
    reference_keypoints_sequence = pickle.load(f)

cap_ref = cv2.VideoCapture(REFERENCE_VIDEO_PATH)
mp_pose = mp.solutions.pose
pose_model = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

def extract_landmarks(frame):
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose_model.process(image_rgb)
    if results.pose_landmarks:
        return np.array([[lm.x, lm.y, lm.z] for lm in results.pose_landmarks.landmark]), results
    return None, results

def calculate_angle(p1, p2, p3):
    v1 = p1 - p2
    v2 = p3 - p2
    dot = np.dot(v1, v2)
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    angle_rad = np.arccos(np.clip(dot / (norm1 * norm2), -1.0, 1.0))
    return np.degrees(angle_rad)

def get_pose_angles(keypoints):
    if keypoints is None or len(keypoints) < 33:
        return None
    mp_lm = mp_pose.PoseLandmark
    angles = {
        'right_elbow': calculate_angle(keypoints[mp_lm.RIGHT_SHOULDER.value], keypoints[mp_lm.RIGHT_ELBOW.value], keypoints[mp_lm.RIGHT_WRIST.value]),
        'right_shoulder': calculate_angle(keypoints[mp_lm.RIGHT_HIP.value], keypoints[mp_lm.RIGHT_SHOULDER.value], keypoints[mp_lm.RIGHT_ELBOW.value]),
        'left_elbow': calculate_angle(keypoints[mp_lm.LEFT_SHOULDER.value], keypoints[mp_lm.LEFT_ELBOW.value], keypoints[mp_lm.LEFT_WRIST.value]),
        'left_shoulder': calculate_angle(keypoints[mp_lm.LEFT_HIP.value], keypoints[mp_lm.LEFT_SHOULDER.value], keypoints[mp_lm.LEFT_ELBOW.value]),
        'right_knee': calculate_angle(keypoints[mp_lm.RIGHT_HIP.value], keypoints[mp_lm.RIGHT_KNEE.value], keypoints[mp_lm.RIGHT_ANKLE.value]),
        'right_hip': calculate_angle(keypoints[mp_lm.RIGHT_SHOULDER.value], keypoints[mp_lm.RIGHT_HIP.value], keypoints[mp_lm.RIGHT_KNEE.value]),
        'left_knee': calculate_angle(keypoints[mp_lm.LEFT_HIP.value], keypoints[mp_lm.LEFT_KNEE.value], keypoints[mp_lm.LEFT_ANKLE.value]),
        'left_hip': calculate_angle(keypoints[mp_lm.LEFT_SHOULDER.value], keypoints[mp_lm.LEFT_HIP.value], keypoints[mp_lm.LEFT_KNEE.value])
    }
    return angles

def compare_poses(ref_keypoints, user_keypoints):
    ref_angles = get_pose_angles(ref_keypoints)
    user_angles = get_pose_angles(user_keypoints)
    if ref_angles is None or user_angles is None:
        return 0.0
    score = 0
    for k in ref_angles:
        diff = abs(ref_angles[k] - user_angles[k])
        similarity = max(0, 1.0 - (diff / 60.0))
        score += similarity
    return score / len(ref_angles)

def main():
    cap_live = cv2.VideoCapture(0)
    total_frames = int(cap_ref.get(cv2.CAP_PROP_FRAME_COUNT))
    ref_count = len(reference_keypoints_sequence)

    # --- Step 1: Get Ready Countdown ---
    cap_ref.set(cv2.CAP_PROP_POS_FRAMES, 0)
    ret_ref, first_ref_frame = cap_ref.read()
    if not ret_ref:
        print("Reference video frame read error.")
        return

    display_ref_frame = cv2.resize(first_ref_frame, (DISPLAY_WINDOW_WIDTH, DISPLAY_WINDOW_HEIGHT))
    cv2.putText(display_ref_frame, INSTRUCTION_MESSAGE, (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,165,255), 2)
    cv2.imshow("Reference Dance", display_ref_frame)

    print("Get ready! Dance comparison will begin in 5 seconds...")
    t_end = time.time() + POSE_WAIT_SECONDS
    while time.time() < t_end:
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cap_live.release()
            cap_ref.release()
            cv2.destroyAllWindows()
            return

    cv2.destroyWindow("Reference Dance")
    print("Dance matching started.")

    # --- Step 2: Continuous Comparison ---
    idx = 0
    final_scores = []

    while idx < ref_count:
        # Reference video frame for display
        cap_ref.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret_ref, ref_frame = cap_ref.read()
        if not ret_ref:
            break
        display_ref_frame = cv2.resize(ref_frame, (DISPLAY_WINDOW_WIDTH, DISPLAY_WINDOW_HEIGHT))
        cv2.imshow("Reference Dance", display_ref_frame)

        # Live capture and pose
        ret_live, live_frame = cap_live.read()
        if not ret_live:
            break
        live_frame = cv2.flip(live_frame, 1)
        user_keypoints, user_results = extract_landmarks(live_frame)
        display_live_frame = cv2.resize(live_frame, (DISPLAY_WINDOW_WIDTH, DISPLAY_WINDOW_HEIGHT))
        if user_results and user_results.pose_landmarks:
            mp_drawing.draw_landmarks(display_live_frame, user_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # ---- Windowed Comparison ----
        best_similarity = 0.0
        for offset in range(-FRAME_WINDOW, FRAME_WINDOW + 1):
            compare_idx = idx + offset
            if 0 <= compare_idx < ref_count:
                ref_keypoints = reference_keypoints_sequence[compare_idx]
                similarity = compare_poses(ref_keypoints, user_keypoints)
                if similarity > best_similarity:
                    best_similarity = similarity

        final_scores.append(best_similarity)

        message = SUCCESS_MESSAGE if best_similarity >= SIMILARITY_THRESHOLD else FAIL_MESSAGE
        color = (0,255,0) if best_similarity >= SIMILARITY_THRESHOLD else (0,0,255)
        cv2.putText(display_live_frame, f"Similarity: {best_similarity:.2f}", (10,50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        cv2.putText(display_live_frame, message, (10,100), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        cv2.imshow("Your Pose", display_live_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        idx += 1

    cap_live.release()
    cap_ref.release()
    cv2.destroyAllWindows()

    avg_score = np.mean(final_scores)
    print(f"\nFinal Average Similarity Score: {avg_score:.2f}")
    if avg_score >= SIMILARITY_THRESHOLD:
        print("✅ Well done! You matched the reference dance.")
    else:
        print("❌ Keep practicing to improve your dance.")

if __name__ == "__main__":
    main()
