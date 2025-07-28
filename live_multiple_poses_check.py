import cv2
import mediapipe as mp
import numpy as np
import pickle
import time # Import time module for a brief pause

# --- Configuration ---
SIMILARITY_THRESHOLD = 0.90 # Set the required similarity to advance
INSTRUCTION_MESSAGE = "Adjust your pose!" # General instruction
SUCCESS_MESSAGE = "Great job! Moving to next pose..."
POSE_HOLD_DURATION = 2 # Seconds to hold the pose after reaching threshold before advancing

# --- Load all reference keypoints and images ---
output_file = "all_reference_keypoints.pkl"
try:
    with open(output_file, "rb") as f:
        all_reference_keypoints_dict = pickle.load(f)
except FileNotFoundError:
    print(f"Error: '{output_file}' not found. Please run save_multiple_references.py first.")
    exit()

# List of images and corresponding keys in the dictionary for display order
# IMPORTANT: These filenames must match the images you used when creating the .pkl file
POSE_SEQUENCE = [
    {"name": "Aramandi", "image": "pose1.jpg"},
    {"name": "Tribhanga", "image": "pose2.jpg"},
    {"name": "Katakamukha", "image": "pose3.jpg"},
    {"name": "Upward Mudra", "image": "pose4.jpg"},
    {"name": "Grounded Pose", "image": "pose5.jpg"},
    # Add more as needed, ensure names match keys in all_reference_keypoints_dict
]

current_pose_index = 0
current_target_pose_name = POSE_SEQUENCE[current_pose_index]["name"]
current_target_keypoints = all_reference_keypoints_dict[current_target_pose_name]
current_target_image_path = POSE_SEQUENCE[current_pose_index]["image"]
current_target_image = cv2.imread(current_target_image_path)

if current_target_image is None:
    print(f"Error: Could not load initial target image: {current_target_image_path}")
    exit()

# --- Initialize MediaPipe Pose ---
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

# --- Helper Functions ---
def extract_landmarks_from_frame(frame):
    results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    if results.pose_landmarks:
        return np.array([[lm.x, lm.y, lm.z] for lm in results.pose_landmarks.landmark]), results.pose_landmarks
    return None, None

def cosine_similarity(a, b):
    a = a.flatten()
    b = b.flatten()
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return np.dot(a, b) / (norm_a * norm_b)

# --- Start webcam ---
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Resize reference image for display
display_width = 400
display_height = int(current_target_image.shape[0] * (display_width / current_target_image.shape[1]))
display_target_image = cv2.resize(current_target_image, (display_width, display_height))

# Variables for automatic advancement
pose_achieved_time = None # To track when the pose threshold was first met

print("Webcam opened. Try to match the pose. Press 'q' to quit.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1) # Flip for natural webcam mirroring

    live_keypoints, live_pose_landmarks_obj = extract_landmarks_from_frame(frame)

    sim = 0.0 # Default similarity
    feedback_message = INSTRUCTION_MESSAGE
    message_color = (0, 165, 255) # Orange for instructions

    if live_keypoints is not None:
        sim = cosine_similarity(current_target_keypoints, live_keypoints)

        if sim >= SIMILARITY_THRESHOLD:
            message_color = (0, 255, 0) # Green for success
            if pose_achieved_time is None: # First time threshold is met for this pose
                pose_achieved_time = time.time()
                feedback_message = f"Perfect! Hold for {POSE_HOLD_DURATION}s..."
            else: # Already holding the pose
                remaining_time = POSE_HOLD_DURATION - (time.time() - pose_achieved_time)
                if remaining_time <= 0:
                    feedback_message = SUCCESS_MESSAGE
                else:
                    feedback_message = f"Perfect! Hold for {remaining_time:.1f}s..."


            # Check if it's time to advance
            if pose_achieved_time is not None and (time.time() - pose_achieved_time) >= POSE_HOLD_DURATION:
                # Reset timer
                pose_achieved_time = None
                
                # Move to next pose
                current_pose_index = (current_pose_index + 1) % len(POSE_SEQUENCE) # Cycle through poses
                current_target_pose_name = POSE_SEQUENCE[current_pose_index]["name"]
                current_target_keypoints = all_reference_keypoints_dict[current_target_pose_name]

                # Load and resize the next target image
                current_target_image_path = POSE_SEQUENCE[current_pose_index]["image"]
                current_target_image = cv2.imread(current_target_image_path)
                if current_target_image is None:
                    print(f"Error: Could not load next target image: {current_target_image_path}. Exiting.")
                    break
                display_target_image = cv2.resize(current_target_image, (display_width, display_height))
                print(f"Moved to next pose: {current_target_pose_name}. Try to match this pose!")
        else:
            feedback_message = INSTRUCTION_MESSAGE
            message_color = (0, 165, 255) # Orange for instructions
            pose_achieved_time = None # Reset timer if pose is no longer held above threshold

        # Draw landmarks on the live frame
        if live_pose_landmarks_obj:
            mp_drawing.draw_landmarks(frame, live_pose_landmarks_obj, mp_pose.POSE_CONNECTIONS)
    else:
        feedback_message = 'Pose not detected'
        message_color = (0, 0, 255) # Red for no detection
        pose_achieved_time = None # Reset timer if pose is not detected

    # Display current pose name
    cv2.putText(frame, f'Target: {current_target_pose_name}', (10, frame.shape[0] - 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2) # Yellow text

    # Display similarity
    cv2.putText(frame, f'Similarity: {sim:.2f}', (10, frame.shape[0] - 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, message_color, 2)

    # Display feedback message
    cv2.putText(frame, feedback_message, (10, frame.shape[0] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, message_color, 2)


    cv2.imshow("Live Pose Comparison", frame) # Window for webcam feed
    cv2.imshow("Target Pose", display_target_image) # Window for displaying the target image

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()