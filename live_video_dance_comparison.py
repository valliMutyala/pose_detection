import cv2
import mediapipe as mp
import numpy as np
import pickle
import time
import os
from ffpyplayer.player import MediaPlayer 

# --- Configuration (GLOBAL CONSTANTS) ---
# Display window settings (adjust these if your screen is very small or large)
DISPLAY_WINDOW_WIDTH = 640 # Target width for both video display windows
# Height will be calculated to maintain aspect ratio

SIMILARITY_THRESHOLD = 0.80 # Your desired similarity to pass a segment
SEGMENT_MASTERY_THRESHOLD_PERCENT = 90 # % of frames in segment that must meet SIMILARITY_THRESHOLD

INSTRUCTION_MESSAGE = "Match the dancer's pose!"
SUCCESS_MESSAGE = "Great job! Hold or prepare for next part!"
NO_POSE_MESSAGE = "Pose not detected in you or reference."
NO_USER_POSE_MESSAGE = "Your pose not detected!"
NO_REF_POSE_MESSAGE = "Reference pose not detected in this frame!"
COMPLETED_ALL_SEGMENTS_MESSAGE = "Congratulations! You mastered the entire dance!"
REFERENCE_KEYPOINTS_FILE="segmented_dance_keypoints.pkl"
SEGMENTS_FOLDER="bharatanatyam_segments"
# --- Load all reference keypoints for segments ---


try:
    all_segments_keypoints = {}
    with open(REFERENCE_KEYPOINTS_FILE, "rb") as f:
        all_segments_keypoints = pickle.load(f)
    print(f"Loaded keypoints for {len(all_segments_keypoints)} dance segments.")
except FileNotFoundError:
    print(f"Error: '{REFERENCE_KEYPOINTS_FILE}' not found. Please run extract_segmented_dance_keypoints.py first.")
    exit()

# Get sorted list of segment filenames to ensure correct order
segment_filenames = sorted(all_segments_keypoints.keys())
if not segment_filenames:
    print("Error: No keypoints found in the loaded file. Exiting.")
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
    
    return landmarks, results

def cosine_similarity(a, b):
    a = a.flatten()
    b = b.flatten()
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return np.dot(a, b) / (norm_a * norm_b)

# --- Start Webcam ---
cap_live = cv2.VideoCapture(0) # For live webcam
if not cap_live.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Get webcam properties for display scaling
live_width = int(cap_live.get(cv2.CAP_PROP_FRAME_WIDTH))
live_height = int(cap_live.get(cv2.CAP_PROP_FRAME_HEIGHT))
live_aspect_ratio = live_width / live_height
calculated_live_display_height = int(DISPLAY_WINDOW_WIDTH / live_aspect_ratio)

# --- Main Segment Loop ---
current_segment_index = 0
while current_segment_index < len(segment_filenames):
    segment_filename = segment_filenames[current_segment_index]
    segment_video_path = os.path.join(SEGMENTS_FOLDER, segment_filename)
    current_segment_keypoints = all_segments_keypoints[segment_filename]

    cap_ref_segment = cv2.VideoCapture(segment_video_path)
    if not cap_ref_segment.isOpened():
        print(f"Error: Could not open segment video '{segment_video_path}'. Skipping this segment.")
        current_segment_index += 1
        continue

    player_segment = MediaPlayer(segment_video_path)

    # Get reference segment video properties for display scaling
    ref_width = int(cap_ref_segment.get(cv2.CAP_PROP_FRAME_WIDTH))
    ref_height = int(cap_ref_segment.get(cv2.CAP_PROP_FRAME_HEIGHT))
    ref_aspect_ratio = ref_width / ref_height
    calculated_ref_display_height = int(DISPLAY_WINDOW_WIDTH / ref_aspect_ratio)

    REF_VIDEO_FPS = cap_ref_segment.get(cv2.CAP_PROP_FPS)
    if REF_VIDEO_FPS == 0: REF_VIDEO_FPS = 30.0 
    FRAME_DELAY_MS = 1000 / REF_VIDEO_FPS 

    current_segment_frame_index = 0
    start_time = time.time() # Reset time for each segment

    frames_correctly_matched_segment = 0
    total_comparable_frames_segment = 0
    
    print(f"\n--- Starting Segment {current_segment_index + 1}/{len(segment_filenames)}: '{segment_filename}' ---")
    print("Try to match this part of the dance!")

    # --- Inner Loop for Current Segment Playback and Comparison ---
    segment_mastered = False
    
    # Initialize ref_frame here before the loop begins its iterations
    ret_ref, ref_frame = cap_ref_segment.read() 
    if not ret_ref:
        print(f"Error: Could not read first frame of segment '{segment_filename}'. Skipping this segment.")
        cap_ref_segment.release()
        player_segment.close_player()
        current_segment_index += 1
        continue # Skip to next segment in outer loop

    while cap_ref_segment.isOpened(): # This loop condition needs to be true based on the initial read
        ret_live, live_frame = cap_live.read()
        
        # Read next ref_frame here for processing THIS iteration
        # Note: ref_frame from initial read is used for first display.
        # This read here is for the *next* frame of the video in the sequence.
        # However, for correct sync, `cap_ref_segment.read()` should be directly
        # tied to the loop's progression based on expected timing.
        # The `ret_ref` check and re-read logic inside `if not ret_ref:` is for looping.

        audio_frame, val = player_segment.get_frame() 

        if not ret_live:
            print("Live webcam feed ended or disconnected.")
            segment_mastered = False # Cannot master if webcam stops
            break # Break inner loop, will lead to outer loop breaking too

        # --- Segment Looping/Completion Logic ---
        # If the video stream for ref_frame has ended (ret_ref from previous read was false)
        # or if the current_segment_frame_index exceeds the keypoints sequence
        if current_segment_frame_index >= len(current_segment_keypoints) or not ret_ref: 
            mastery_percentage = 0
            if total_comparable_frames_segment > 0:
                mastery_percentage = (frames_correctly_matched_segment / total_comparable_frames_segment) * 100

            print(f"  Segment '{segment_filename}' finished. Mastery: {mastery_percentage:.2f}% (Target: {SEGMENT_MASTERY_THRESHOLD_PERCENT}%)")

            if mastery_percentage >= SEGMENT_MASTERY_THRESHOLD_PERCENT:
                print(f"  Segment {current_segment_index + 1} mastered! Moving to next segment...")
                segment_mastered = True
                break # Exit inner loop, move to next segment

            else:
                print(f"  Segment {current_segment_index + 1} not yet mastered. Looping for more practice!")
                
                # Reset segment video and audio
                cap_ref_segment.set(cv2.CAP_PROP_POS_FRAMES, 0)
                player_segment.close_player() 
                player_segment = MediaPlayer(segment_video_path) 
                
                current_segment_frame_index = 0
                start_time = time.time() # Reset timer for new loop
                frames_correctly_matched_segment = 0 # Reset mastery tracking
                total_comparable_frames_segment = 0
                time.sleep(0.5) 
                
                # Crucial: Re-read first frame after reset for the NEXT iteration's display
                ret_ref, ref_frame = cap_ref_segment.read() 
                if not ret_ref:
                    print(f"Error: Could not read first frame after reset for segment {segment_filename}. Skipping.")
                    break # Break inner loop, will move to next segment or end if last


        # Safeguard: ensure frames are valid before processing/displaying
        if ref_frame is None or ref_frame.size == 0:
            print(f"  Warning: Invalid reference frame at index {current_segment_frame_index}. Skipping.")
            current_segment_frame_index += 1
            # Skip if ref_frame is bad, but still show live frame if possible
            # Need to re-read ref_frame for next iteration if skipping this one
            ret_ref, ref_frame = cap_ref_segment.read() # Try reading next frame
            continue 
        if live_frame is None or live_frame.size == 0:
            print("  Warning: Invalid live frame. Skipping.")
            continue
        
        # Flip live frame for natural webcam mirroring
        live_frame = cv2.flip(live_frame, 1)

        # --- Resize frames for consistent display ---
        display_live_frame = cv2.resize(live_frame, (DISPLAY_WINDOW_WIDTH, calculated_live_display_height), interpolation=cv2.INTER_AREA)
        display_ref_frame = cv2.resize(ref_frame, (DISPLAY_WINDOW_WIDTH, calculated_ref_display_height), interpolation=cv2.INTER_AREA)

        # --- Extract Live Pose (from original resolution live_frame) ---
        live_keypoints, live_pose_landmarks_obj = extract_landmarks_from_frame(live_frame, pose) 

        # --- Get Corresponding Reference Pose Keypoints for Comparison ---
        current_ref_keypoints_for_comparison = None
        # Ensure index is within bounds of keypoints sequence
        if current_segment_frame_index < len(current_segment_keypoints):
            current_ref_keypoints_for_comparison = current_segment_keypoints[current_segment_frame_index]
        
        # --- Comparison Logic & Feedback ---
        sim = 0.0
        feedback_message = NO_POSE_MESSAGE
        message_color = (0, 0, 255) 

        if live_keypoints is not None and current_ref_keypoints_for_comparison is not None:
            total_comparable_frames_segment += 1 
            sim = cosine_similarity(current_ref_keypoints_for_comparison, live_keypoints)

            if sim >= SIMILARITY_THRESHOLD:
                feedback_message = SUCCESS_MESSAGE
                message_color = (0, 255, 0) 
                frames_correctly_matched_segment += 1 
            else:
                feedback_message = INSTRUCTION_MESSAGE
                message_color = (0, 165, 255) 
        elif live_keypoints is None and current_ref_keypoints_for_comparison is not None:
            feedback_message = NO_USER_POSE_MESSAGE
            message_color = (0, 0, 255) 
        elif live_keypoints is not None and current_ref_keypoints_for_comparison is None:
            # This means reference keypoints are missing for this specific frame
            # (e.g., MediaPipe failed during extraction for that frame)
            feedback_message = NO_REF_POSE_MESSAGE
            message_color = (0, 0, 255) 


        # --- Display Info on Live Feed ---
        progress_text = f"Segment {current_segment_index + 1}/{len(segment_filenames)} | Frame: {current_segment_frame_index}/{len(current_segment_keypoints)}"
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
        cv2.imshow("Reference Dance Segment", display_ref_frame) 

        # --- Synchronization Logic for Smooth Playback ---
        current_time_loop = time.time() 
        elapsed_time_since_segment_start = (current_time_loop - start_time) * 1000 
        expected_time_for_frame = current_segment_frame_index * FRAME_DELAY_MS
        
        wait_time_ms = int(expected_time_for_frame - elapsed_time_since_segment_start)
        
        key = cv2.waitKey(max(1, wait_time_ms)) & 0xFF 

        if key == ord('q'):
            cap_live.release()
            cap_ref_segment.release()
            player_segment.close_player()
            cv2.destroyAllWindows()
            exit() 
        
        current_segment_frame_index += 1 

    # --- End of Inner Loop (Segment Completion/Looping) ---
    cap_ref_segment.release() # Release this segment's video capture
    player_segment.close_player() # Close this segment's audio player
    
    # If the segment was mastered, the outer loop will increment current_segment_index
    # If not mastered, it stays the same, and the outer loop will re-enter this segment

# --- After all segments are completed ---
print(f"\n--- All {len(segment_filenames)} Dance Segments Completed! ---")
final_message_frame = np.zeros((calculated_live_display_height, DISPLAY_WINDOW_WIDTH, 3), dtype=np.uint8)
cv2.putText(final_message_frame, COMPLETED_ALL_SEGMENTS_MESSAGE, (10, calculated_live_display_height // 2), 
            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
cv2.imshow("Final Message", final_message_frame)
cv2.waitKey(0) # Wait indefinitely until any key is pressed

# --- Clean Up ---
cap_live.release()
cv2.destroyAllWindows()