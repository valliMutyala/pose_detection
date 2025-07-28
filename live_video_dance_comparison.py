import cv2
import mediapipe as mp
import numpy as np
import pickle
import time
from ffpyplayer.player import MediaPlayer 

# --- Configuration ---
# IMPORTANT: These paths must match the video and the generated .pkl file
REFERENCE_VIDEO_PATH = "bharatanatyam1.mp4" 
REFERENCE_KEYPOINTS_FILE = "reference_dance_keypoints.pkl" 

SIMILARITY_THRESHOLD = 0.85 # Adjust as needed (0.0 to 1.0)
INSTRUCTION_MESSAGE = "Match the dancer's pose!"
SUCCESS_MESSAGE = "Great job! Keep going!"
NO_POSE_MESSAGE = "Pose not detected in you or reference."
NO_USER_POSE_MESSAGE = "Your pose not detected!"
NO_REF_POSE_MESSAGE = "Reference pose not detected in this frame!"
DANCE_MASTERY_THRESHOLD_PERCENT = 90 

# Display window settings (adjust these if your screen is very small or large)
DISPLAY_WINDOW_WIDTH = 640 # Target width for both video display windows
# Height will be calculated to maintain aspect ratio

# --- Load pre-extracted reference keypoints from the video ---
try:
    reference_dance_keypoints_sequence = []
    with open(REFERENCE_KEYPOINTS_FILE, "rb") as f:
        reference_dance_keypoints_sequence = pickle.load(f)
    print(f"Loaded {len(reference_dance_keypoints_sequence)} reference poses from '{REFERENCE_KEYPOINTS_FILE}'.")
except FileNotFoundError:
    print(f"Error: '{REFERENCE_KEYPOINTS_FILE}' not found. Please run extract_reference_dance_video_keypoints.py first.")
    exit()

# --- Initialize MediaPipe Pose for Live Feed ---
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils 

# --- Helper Functions ---
def extract_landmarks_from_frame(frame, pose_model):
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose_model.process(image_rgb)
    
    landmarks = None
    if results.pose_landmarks:
        landmarks = np.array([[lm.x, lm.y, lm.z] for lm in results.pose_landmarks.landmark])
    
    return landmarks, results # Return results object for drawing

def cosine_similarity(a, b):
    a = a.flatten()
    b = b.flatten()
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return np.dot(a, b) / (norm_a * norm_b)

# --- Start Webcam and Reference Video ---
cap_live = cv2.VideoCapture(0) # For live webcam
cap_ref = cv2.VideoCapture(REFERENCE_VIDEO_PATH) # For playing the reference video

player = MediaPlayer(REFERENCE_VIDEO_PATH) # Initialize ffpyplayer for audio

if not cap_live.isOpened():
    print("Error: Could not open webcam.")
    exit()
if not cap_ref.isOpened():
    print(f"Error: Could not open reference video file '{REFERENCE_VIDEO_PATH}'.")
    exit()

# Get properties for both live and reference videos for consistent display scaling
# Live Webcam properties
live_width = int(cap_live.get(cv2.CAP_PROP_FRAME_WIDTH))
live_height = int(cap_live.get(cv2.CAP_PROP_FRAME_HEIGHT))
live_aspect_ratio = live_width / live_height
calculated_live_display_height = int(DISPLAY_WINDOW_WIDTH / live_aspect_ratio)

# Reference Video properties
ref_width = int(cap_ref.get(cv2.CAP_PROP_FRAME_WIDTH))
ref_height = int(cap_ref.get(cv2.CAP_PROP_FRAME_HEIGHT))
ref_aspect_ratio = ref_width / ref_height
calculated_ref_display_height = int(DISPLAY_WINDOW_WIDTH / ref_aspect_ratio)

REF_VIDEO_FPS = cap_ref.get(cv2.CAP_PROP_FPS)
if REF_VIDEO_FPS == 0: 
    REF_VIDEO_FPS = 30.0 
FRAME_DELAY_MS = 1000 / REF_VIDEO_FPS 

current_ref_frame_index = 0
start_time = time.time() 

frames_correctly_matched = 0
total_comparable_frames = 0 

print("Starting dance comparison. Press 'q' to quit.")
print("Try to match the dance as closely as possible.")

# Initialize ref_frame with a valid frame before the loop starts
# This ensures ref_frame is never None or empty at the first imshow call.
initial_ret_ref, ref_frame = cap_ref.read()
if not initial_ret_ref:
    print("Error: Could not read first frame of reference video. Exiting.")
    cap_ref.release()
    cap_live.release()
    player.close_player()
    cv2.destroyAllWindows()
    exit()

while cap_live.isOpened():
    ret_live, live_frame = cap_live.read()
    
    # Read ref_frame immediately inside the loop for each iteration
    ret_ref, ref_frame = cap_ref.read() 
    audio_frame, val = player.get_frame() 

    if not ret_live:
        print("Live webcam feed ended or disconnected.")
        break
    
    # --- Looping Logic for Reference Video ---
    if not ret_ref: # End of video or error reading frame
        mastery_percentage = 0
        if total_comparable_frames > 0:
            mastery_percentage = (frames_correctly_matched / total_comparable_frames) * 100

        print(f"\n--- Dance Sequence Finished (Loop {current_ref_frame_index // len(reference_dance_keypoints_sequence) + 1}) ---")
        print(f"Your Mastery: {mastery_percentage:.2f}% (Target: {DANCE_MASTERY_THRESHOLD_PERCENT}%)")

        if mastery_percentage >= DANCE_MASTERY_THRESHOLD_PERCENT:
            print("Congratulations! You've mastered this dance sequence!")
            feedback_message = "DANCE MASTERED! Press 'q' to quit."
            message_color = (0, 255, 255) # Cyan for mastery
            
            # Display final message and wait for 'q'
            while True:
                ret_live_final, live_frame_final = cap_live.read()
                if not ret_live_final: break
                live_frame_final = cv2.flip(live_frame_final, 1) 
                
                # Resize final live frame for display
                display_live_final_frame = cv2.resize(live_frame_final, (DISPLAY_WINDOW_WIDTH, calculated_live_display_height), interpolation=cv2.INTER_AREA)

                cv2.putText(display_live_final_frame, f'Mastery: {mastery_percentage:.2f}%', (10, display_live_final_frame.shape[0] // 2 - 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, message_color, 2)
                cv2.putText(display_live_final_frame, feedback_message, (10, display_live_final_frame.shape[0] // 2 + 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, message_color, 2)
                
                # Show a blank frame or the last valid ref_frame if you want
                display_final_ref_frame = ref_frame if ref_frame is not None and ref_frame.size > 0 else np.zeros((calculated_ref_display_height, DISPLAY_WINDOW_WIDTH, 3), dtype=np.uint8)
                cv2.imshow("Reference Dance Video", display_final_ref_frame) 
                cv2.imshow("Your Live Performance", display_live_final_frame)

                if cv2.waitKey(100) & 0xFF == ord('q'): 
                    break
            break # Exit main loop after mastery

        else:
            print("Not yet mastered. Looping the dance for more practice!")
            feedback_message = "Looping for more practice!"
            message_color = (0, 100, 255) # Darker orange for looping message
            
            # Reset video to start
            cap_ref.set(cv2.CAP_PROP_POS_FRAMES, 0)
            player.close_player() 
            player = MediaPlayer(REFERENCE_VIDEO_PATH) 
            current_ref_frame_index = 0
            start_time = time.time() 
            frames_correctly_matched = 0 
            total_comparable_frames = 0
            time.sleep(0.5) # Brief pause before restarting.

            # CRUCIAL FIX: Re-read the first frame *immediately* after resetting the video
            ret_ref, ref_frame = cap_ref.read()
            if not ret_ref:
                print("Error: Could not read first frame after video reset. Exiting.")
                break 
            
    # Ensure ref_frame is valid before proceeding with imshow later.
    # This also applies to live_frame after flipping.
    if ref_frame is None or ref_frame.size == 0:
        print("Warning: Reference frame is empty or invalid. Skipping this loop iteration.")
        current_ref_frame_index += 1 
        continue 
    if live_frame is None or live_frame.size == 0:
        print("Warning: Live frame is empty or invalid. Skipping this loop iteration.")
        continue


    # --- Resize frames for consistent display ---
    display_live_frame = cv2.resize(live_frame, (DISPLAY_WINDOW_WIDTH, calculated_live_display_height), interpolation=cv2.INTER_AREA)
    display_ref_frame = cv2.resize(ref_frame, (DISPLAY_WINDOW_WIDTH, calculated_ref_display_height), interpolation=cv2.INTER_AREA)

    # --- Extract Live Pose (from original resolution live_frame, then draw on display_live_frame) ---
    # MediaPipe works best on original resolution, then we draw on scaled for display
    live_keypoints, live_pose_landmarks_obj = extract_landmarks_from_frame(live_frame, pose) 

    # --- Get Corresponding Reference Pose Keypoints for Comparison ---
    current_ref_keypoints_for_comparison = None
    if current_ref_frame_index < len(reference_dance_keypoints_sequence):
        current_ref_keypoints_for_comparison = reference_dance_keypoints_sequence[current_ref_frame_index]
    
    # --- Comparison Logic & Feedback ---
    sim = 0.0
    feedback_message = NO_POSE_MESSAGE
    message_color = (0, 0, 255) 

    if live_keypoints is not None and current_ref_keypoints_for_comparison is not None:
        total_comparable_frames += 1 
        sim = cosine_similarity(current_ref_keypoints_for_comparison, live_keypoints)

        if sim >= SIMILARITY_THRESHOLD:
            feedback_message = SUCCESS_MESSAGE
            message_color = (0, 255, 0) 
            frames_correctly_matched += 1 
        else:
            feedback_message = INSTRUCTION_MESSAGE
            message_color = (0, 165, 255) 
    elif live_keypoints is None and current_ref_keypoints_for_comparison is not None:
         feedback_message = NO_USER_POSE_MESSAGE
         message_color = (0, 0, 255) 
    elif live_keypoints is not None and current_ref_keypoints_for_comparison is None:
         feedback_message = NO_REF_POSE_MESSAGE
         message_color = (0, 0, 255) 


    # --- Display Info on Live Feed ---
    progress_text = f"Frame: {current_ref_frame_index}/{len(reference_dance_keypoints_sequence)}"
    cv2.putText(display_live_frame, progress_text, (10, display_live_frame.shape[0] - 140),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1) 

    cv2.putText(display_live_frame, f'Similarity: {sim:.2f}', (10, display_live_frame.shape[0] - 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, message_color, 2)
    cv2.putText(display_live_frame, feedback_message, (10, display_live_frame.shape[0] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, message_color, 2)

    # Draw landmarks on the RESIZED live frame
    if live_pose_landmarks_obj and live_pose_landmarks_obj.pose_landmarks: 
        mp_drawing.draw_landmarks(display_live_frame, live_pose_landmarks_obj.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    # --- Display Video Windows ---
    cv2.imshow("Your Live Performance", display_live_frame)
    cv2.imshow("Reference Dance Video", display_ref_frame) 

    # --- Synchronization Logic for Smooth Playback ---
    current_time = time.time()
    elapsed_time_since_start = (current_time - start_time) * 1000 
    expected_time_for_frame = current_ref_frame_index * FRAME_DELAY_MS
    
    wait_time_ms = int(expected_time_for_frame - elapsed_time_since_start)
    
    key = cv2.waitKey(max(1, wait_time_ms)) & 0xFF 

    if key == ord('q'):
        break
    
    current_ref_frame_index += 1 

# --- Clean Up ---
cap_live.release()
cap_ref.release()
player.close_player() 
cv2.destroyAllWindows()