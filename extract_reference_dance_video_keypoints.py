import cv2
import mediapipe as mp
import numpy as np
import pickle
import os

# --- Configuration ---
REFERENCE_VIDEO_PATH = "bharatanatyam1.mp4" 
OUTPUT_KEYPOINTS_FILE = "reference_dance_keypoints.pkl" 
DISPLAY_VIDEO_DURING_EXTRACTION = True 

# Display window settings (adjust these if your video is very large/small)
DISPLAY_WIDTH = 640 # Recommended width for display
DISPLAY_HEIGHT = 480 # Recommended height for display (maintains aspect ratio if possible)


# --- Initialize MediaPipe Pose ---
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils 

# --- Function to extract pose landmarks from a frame ---
def extract_landmarks_from_frame(frame, pose_model):
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose_model.process(image_rgb)
    
    landmarks = None
    if results.pose_landmarks:
        landmarks = np.array([[lm.x, lm.y, lm.z] for lm in results.pose_landmarks.landmark])
    
    return landmarks, results

# --- Main script execution ---
def process_reference_video_for_keypoints(video_path, output_file, display_video):
    if not os.path.exists(video_path):
        print(f"Error: Reference video file not found at '{video_path}'")
        return

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: Could not open reference video file '{video_path}'")
        # Try different backends if video doesn't open (less common but happens)
        # cap = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG) # Example: force FFmpeg backend
        return

    # Get original video dimensions to maintain aspect ratio
    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Calculate aspect ratio
    aspect_ratio = original_width / original_height
    
    # Adjust DISPLAY_HEIGHT based on DISPLAY_WIDTH to maintain aspect ratio
    calculated_display_height = int(DISPLAY_WIDTH / aspect_ratio)
    
    # If the calculated height is too small or too large, you might want to constrain it
    if calculated_display_height > 1000 or calculated_display_height < 200: # Example constraints
        calculated_display_height = DISPLAY_HEIGHT # Fallback to default if aspect ratio results in weird size
        print(f"Warning: Calculated display height ({calculated_display_height}) is unusual. Using default {DISPLAY_HEIGHT}.")


    all_video_keypoints_sequence = []
    frame_count = 0

    print(f"Starting to extract keypoints from reference video: {video_path}")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break # End of video or error reading frame

        frame_count += 1
        landmarks, results = extract_landmarks_from_frame(frame, pose)
        
        all_video_keypoints_sequence.append(landmarks) 
        
        if display_video:
            # Resize the frame for consistent display
            # Use cv2.INTER_AREA for shrinking, cv2.INTER_LINEAR for zooming
            display_frame = cv2.resize(frame, (DISPLAY_WIDTH, calculated_display_height), interpolation=cv2.INTER_AREA)

            # Draw landmarks on the RESIZED frame
            if results and results.pose_landmarks: 
                # Need to convert normalized landmarks back to absolute pixels for drawing on resized frame
                # This is already handled internally by mp_drawing for display purposes when it gets the results obj.
                mp_drawing.draw_landmarks(display_frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            
            cv2.imshow('Reference Video Extraction', display_frame)

        # Use waitKey(0) to step through frames manually for detailed inspection,
        # or a higher number like 10-30 for slower, but continuous playback.
        # For actual video speed, keep it at 1, but be aware of skipping if processing is slow.
        key = cv2.waitKey(1) & 0xFF # Keep at 1 for close to real-time (will skip frames if processing is slow)
        # key = cv2.waitKey(0) & 0xFF # Uncomment for manual frame-by-frame (press any key for next frame)

        if key == ord('q'):
            print("Reference video extraction interrupted by user.")
            break

    cap.release()
    cv2.destroyAllWindows()

    if all_video_keypoints_sequence:
        with open(output_file, "wb") as f:
            pickle.dump(all_video_keypoints_sequence, f)
        print(f"Successfully extracted and saved keypoints for {len(all_video_keypoints_sequence)} frames to '{output_file}'.")
    else:
        print("No frames processed or no poses detected in the reference video. No keypoints saved.")

# Run the process
if __name__ == "__main__":
    process_reference_video_for_keypoints(REFERENCE_VIDEO_PATH, OUTPUT_KEYPOINTS_FILE, DISPLAY_VIDEO_DURING_EXTRACTION)