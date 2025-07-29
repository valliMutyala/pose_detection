import cv2
import mediapipe as mp
import numpy as np
import pickle
import time
import os

# --- Configuration ---
SIMILARITY_THRESHOLD = 0.90
POSE_HOLD_DURATION = 2
INSTRUCTION_MESSAGE = "Adjust your pose!"
SUCCESS_MESSAGE = "Great job! Moving to next pose..."

# --- Load reference keypoints ---
REFERENCE_FILE = "all_reference_keypoints.pkl"
if not os.path.exists(REFERENCE_FILE):
    print(f"❌ Error: '{REFERENCE_FILE}' not found. Please run save_multiple_references.py first.")
    exit()

with open(REFERENCE_FILE, "rb") as f:
    all_reference_keypoints_dict = pickle.load(f)

# --- Pose sequence ---
POSE_SEQUENCE = [
    {"name": "Aramandi", "image": "pose1.jpg"},
    {"name": "Tribhanga", "image": "pose2.jpg"},
    {"name": "Katakamukha", "image": "pose3.jpg"},
    {"name": "Upward Mudra", "image": "pose4.jpg"},
    {"name": "Grounded Pose", "image": "pose5.jpg"},
]

# --- Initialize Pose Detector ---
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

def extract_landmarks_from_frame(frame):
    results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    if results.pose_landmarks:
        keypoints = np.array([[lm.x, lm.y, lm.z] for lm in results.pose_landmarks.landmark])
        return keypoints, results.pose_landmarks
    return None, None

def cosine_similarity(a, b):
    if a is None or b is None or a.shape != b.shape:
        return 0.0
    a = a.flatten()
    b = b.flatten()
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return np.dot(a, b) / (norm_a * norm_b)

# --- Load initial target pose ---
def load_target_pose(index):
    pose_name = POSE_SEQUENCE[index]["name"]
    image_path = POSE_SEQUENCE[index]["image"]
    keypoints = all_reference_keypoints_dict.get(pose_name, None)
    image = cv2.imread(image_path)
    if keypoints is None or image is None:
        print(f"❌ Failed to load data for pose: {pose_name}. Check image and keypoint file.")
        return None, None, None
    image_resized = cv2.resize(image, (400, int(400 * image.shape[0] / image.shape[1])))
    return pose_name, keypoints, image_resized

# --- Initialize pose ---
current_pose_index = 0
pose_name, target_keypoints, target_image = load_target_pose(current_pose_index)
if target_keypoints is None:
    exit()

# --- Webcam ---
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Error: Could not open webcam.")
    exit()

pose_achieved_time = None
print("📷 Webcam open. Match the pose! Press 'q' to quit.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)

    live_keypoints, live_landmarks = extract_landmarks_from_frame(frame)
    similarity = cosine_similarity(target_keypoints, live_keypoints)

    if live_landmarks:
        mp_drawing.draw_landmarks(frame, live_landmarks, mp_pose.POSE_CONNECTIONS)

    feedback = INSTRUCTION_MESSAGE
    color = (0, 165, 255)  # Orange default

    if live_keypoints is not None:
        if similarity >= SIMILARITY_THRESHOLD:
            if pose_achieved_time is None:
                pose_achieved_time = time.time()
            elapsed = time.time() - pose_achieved_time
            remaining = POSE_HOLD_DURATION - elapsed
            if remaining <= 0:
                feedback = SUCCESS_MESSAGE
                color = (0, 255, 0)
                current_pose_index = (current_pose_index + 1) % len(POSE_SEQUENCE)
                pose_name, target_keypoints, target_image = load_target_pose(current_pose_index)
                pose_achieved_time = None
                time.sleep(0.8)
                print(f"➡️ Next pose: {pose_name}")
            else:
                feedback = f"Perfect! Hold for {remaining:.1f}s..."
                color = (0, 255, 0)
        else:
            pose_achieved_time = None
    else:
        feedback = "Pose not detected"
        color = (0, 0, 255)

    # UI Overlays
    h = frame.shape[0]
    cv2.putText(frame, f'Target: {pose_name}', (10, h - 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
    cv2.putText(frame, f'Similarity: {similarity:.2f}', (10, h - 60), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    cv2.putText(frame, feedback, (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    # Display frames
    cv2.imshow("Live Pose Comparison", frame)
    cv2.imshow("Target Pose", target_image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
