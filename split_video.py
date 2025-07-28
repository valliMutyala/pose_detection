from moviepy.editor import VideoFileClip
import os
import math

def split_video_into_segments(input_video_path, segment_duration_seconds=5, output_folder="split_videos"):
    """
    Splits a video into shorter segments of a specified duration, including audio.

    Args:
        input_video_path (str): The path to the long video file.
        segment_duration_seconds (int): The duration of each output segment in seconds.
        output_folder (str): The name of the folder where the segments will be saved.
    """
    if not os.path.exists(input_video_path):
        print(f"Error: Input video not found at '{input_video_path}'")
        return

    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Created output folder: {output_folder}")

    try:
        video = VideoFileClip(input_video_path)
        total_duration = video.duration # Get total duration in seconds
        
        # Get base filename and extension
        base_name = os.path.splitext(os.path.basename(input_video_path))[0]
        ext = os.path.splitext(input_video_path)[1]

        num_segments = math.ceil(total_duration / segment_duration_seconds)

        print(f"Total video duration: {total_duration:.2f} seconds")
        print(f"Splitting into {num_segments} segments of {segment_duration_seconds} seconds each.")

        for i in range(num_segments):
            start_time = i * segment_duration_seconds
            end_time = min((i + 1) * segment_duration_seconds, total_duration)

            # Ensure the segment has some duration
            if start_time >= end_time:
                continue

            output_filename = os.path.join(
                output_folder,
                f"{base_name}_part{i+1:03d}{ext}" # e.g., bharatanatyam_part001.mp4
            )

            print(f"Processing segment {i+1}: {start_time:.2f}s to {end_time:.2f}s -> '{output_filename}'")
            
            # Subclip and write to file
            # By default, write_videofile includes audio if the original clip has it.
            # We explicitly set codec and audio_codec to ensure compatibility.
            # Adding fps=video.fps ensures original frame rate is maintained.
            segment = video.subclip(start_time, end_time)
            segment.write_videofile(output_filename, codec="libx264", audio_codec="aac", fps=video.fps)
            
        print("\nVideo splitting complete!")

    except Exception as e:
        print(f"An error occurred during video splitting: {e}")
        print("Please ensure FFmpeg is correctly installed and its 'bin' directory is in your system's PATH.")
        print("You can verify FFmpeg installation by typing 'ffmpeg -version' in a new terminal.")

# --- Example Usage ---
if __name__ == "__main__":
    # IMPORTANT: Set your input video path here. 
    # Ensure 'bharatanatyam.mp4' is in the same directory as this script.
    input_video = "bharatanatyam.mp4" # Your long video file
    
    # Set the desired segment duration
    segment_duration = 5 # seconds

    # Set the output folder name
    output_folder_name = "bharatanatyam_segments" # New folder for segments

    split_video_into_segments(input_video, segment_duration, output_folder_name)