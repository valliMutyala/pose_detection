from moviepy.editor import VideoFileClip
import os

VIDEO_PATH = "videoplayback.mp4"
OUTPUT_PREFIX = "bharatanatyam1_part"
OUTPUT_FOLDER = "bharatanatyam_segments"
CHUNK_DURATION = 8  # seconds

def clear_old_segments(folder):
    for filename in os.listdir(folder):
        if filename.startswith(OUTPUT_PREFIX):
            os.remove(os.path.join(folder, filename))

def split_video(video_path, output_folder, chunk_duration=8):
    clip = VideoFileClip(video_path)
    duration = clip.duration
    total_parts = int(duration // chunk_duration + (1 if duration % chunk_duration else 0))

    print(f"Splitting '{video_path}' ({duration:.2f}s) into {total_parts} parts...")

    for i in range(total_parts):
        start = i * chunk_duration
        end = min((i + 1) * chunk_duration, duration)
        part = clip.subclip(start, end)

        filename = f"{OUTPUT_PREFIX}{str(i + 1).zfill(3)}.mp4"
        filepath = os.path.join(output_folder, filename)

        print(f"Writing: {filepath} [{start:.2f}s to {end:.2f}s]")
        part.write_videofile(filepath, codec="libx264", audio_codec="aac", verbose=False, logger=None)

    print("✅ Done splitting!")

if __name__ == "__main__":
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)
    
    clear_old_segments(OUTPUT_FOLDER)
    split_video(VIDEO_PATH, OUTPUT_FOLDER, CHUNK_DURATION)
