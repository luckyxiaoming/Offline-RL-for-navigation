import h5py
import cv2
import numpy as np
import glob
import os

def h5_to_video(h5_path, output_video_path, fps=30):
    print(f"Processing {h5_path} -> {output_video_path}")
    
    try:
        with h5py.File(h5_path, 'r') as f:
            # Check for image topic
            # Based on previous context, topic is 'go2_camera'
            if 'go2_camera' not in f:
                print(f"No 'go2_camera' group in {h5_path}")
                return

            # The data field for CompressedImage is usually 'data'
            if 'data' not in f['go2_camera']:
                 print(f"No 'data' dataset in go2_camera")
                 return
                 
            images_dset = f['go2_camera']['data']
            
            # Try to get timestamps for FPS calculation
            timestamps = None
            if 'header' in f['go2_camera'] and 'stamp' in f['go2_camera']['header']:
                stamp_group = f['go2_camera']['header']['stamp']
                if 'sec' in stamp_group and 'nanosec' in stamp_group:
                    sec = stamp_group['sec'][:]
                    nanosec = stamp_group['nanosec'][:]
                    timestamps = sec + nanosec * 1e-9
            
            # Calculate FPS from timestamps if possible
            if timestamps is not None and len(timestamps) > 1:
                duration = timestamps[-1] - timestamps[0]
                if duration > 0:
                    calc_fps = len(timestamps) / duration
                    print(f"  Calculated FPS: {calc_fps:.2f}")
                    # Use calculated FPS if reasonable, else default
                    if 1 < calc_fps < 120:
                        fps = calc_fps
            
            n_frames = len(images_dset)
            if n_frames == 0:
                print("  No frames found")
                return

            print(f"  Found {n_frames} frames")

            # Decode first frame to get size
            # Handle vlen dataset which returns numpy array
            first_data = images_dset[0]
            if isinstance(first_data, np.ndarray):
                first_img_bytes = first_data
            else:
                first_img_bytes = np.frombuffer(first_data, dtype=np.uint8)
                
            first_img = cv2.imdecode(first_img_bytes, cv2.IMREAD_COLOR)
            
            if first_img is None:
                print("  Failed to decode first image")
                return
                
            height, width, layers = first_img.shape
            size = (width, height)
            print(f"  Video resolution: {width}x{height}")
            
            # Initialize VideoWriter
            # mp4v is a safe bet for .mp4
            fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
            out = cv2.VideoWriter(output_video_path, fourcc, fps, size)
            
            for i in range(n_frames):
                data = images_dset[i]
                if isinstance(data, np.ndarray):
                    img_bytes = data
                else:
                    img_bytes = np.frombuffer(data, dtype=np.uint8)
                    
                img = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)
                
                if img is not None:
                    out.write(img)
                else:
                    print(f"  Warning: Could not decode frame {i}")
                    
            out.release()
            print(f"  Saved {output_video_path}")
            
    except Exception as e:
        print(f"Error processing {h5_path}: {e}")

def main():
    h5_files = sorted(glob.glob("*.h5"))
    if not h5_files:
        print("No .h5 files found.")
        return

    for h5_file in h5_files:
        video_name = h5_file.replace(".h5", ".mp4")
        h5_to_video(h5_file, video_name)

if __name__ == "__main__":
    main()
