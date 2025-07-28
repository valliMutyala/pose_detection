import cv2
import mediapipe as mp
import numpy as np
import pickle
import os

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True)

def extract_landmarks(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image from {image_path}")
        return None

    results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    if results.pose_landmarks:
        return np.array([[lm.x, lm.y, lm.z] for lm in results.pose_landmarks.landmark])
    return None

# --- Configuration ---
# Create a dictionary to store all reference poses
all_reference_poses = {}

# List of image files and their corresponding pose names
# Make sure these images are in the same directory as this script, or provide full paths
reference_images = {
    "Aramandi": "pose1.jpg",  # Rename reference_pose.jpeg or add new ones
    "Tribhanga": "pose2.jpg",
    "Katakamukha": "pose3.jpg",
    "Upward Mudra": "pose4.jpg",
    "Grounded Pose": "pose5.jpg",
}

# Make sure you have standing_pose.jpeg, sitting_pose.jpeg, squat_pose.jpeg in your folder!

output_file = "all_reference_keypoints.pkl" # New output file

print("Starting to process reference poses...")

for pose_name, image_filename in reference_images.items():
    image_path = os.path.join(".", image_filename) # Assumes images are in the current directory
    print(f"Processing '{pose_name}' from '{image_path}'...")
    
    landmarks = extract_landmarks(image_path)

    if landmarks is not None:
        all_reference_poses[pose_name] = landmarks
        print(f"Successfully extracted landmarks for '{pose_name}'.")
    else:
        print(f"No pose detected in the image for '{pose_name}'.")

if all_reference_poses:
    with open(output_file, "wb") as f:
        pickle.dump(all_reference_poses, f)
    print(f"\nAll reference keypoints saved successfully to '{output_file}'.")
else:
    print("\nNo reference poses were saved. Check image paths and ensure poses are detectable.")