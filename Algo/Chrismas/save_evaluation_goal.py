import h5py
import numpy as np
import cv2
import os

def save_evaluation_goals():
    # Configuration of files and indices
    tasks = {
        'bolt_nav_01.h5': [704, 1264, 1435, 1700],
        'bolt_nav_02.h5': [243, 300, 414, 951, 964, 1424, 1537],
        'bolt_nav_05.h5': [316, 1645, 1606],
        'bolt_nav_06.h5': [256, 1182]
    }

    output_h5_path = 'evaluation_goals.h5'
    output_img_dir = 'goal_images'
    
    # Create output directory for images
    os.makedirs(output_img_dir, exist_ok=True)
    
    all_observations = []
    all_features = []
    all_stds = []
    
    print(f"Starting extraction...")
    
    for filename, indices in tasks.items():
        if not os.path.exists(filename):
            print(f"Warning: {filename} not found. Skipping.")
            continue
            
        print(f"Processing {filename} with indices {indices}")
        
        try:
            with h5py.File(filename, 'r') as f:
                # Check for required datasets
                if 'observations' not in f or 'features' not in f or 'stds' not in f:
                    print(f"  Missing datasets in {filename}. Available keys: {list(f.keys())}")
                    continue
                
                # Read data
                obs_dset = f['observations']
                feat_dset = f['features']
                std_dset = f['stds']
                
                for idx in indices:
                    try:
                        obs = obs_dset[idx]
                        feat = feat_dset[idx]
                        std = std_dset[idx]
                        
                        all_observations.append(obs)
                        all_features.append(feat)
                        all_stds.append(std)
                        
                        # Save image
                        # Naming convention: filename_index.png
                        img_name = f"{os.path.splitext(filename)[0]}_{idx}.png"
                        img_path = os.path.join(output_img_dir, img_name)
                        
                        # Check if obs is valid image
                        if obs.ndim == 3 and obs.shape[2] == 3:
                            # OpenCV uses BGR, but ML datasets are often RGB.
                            # If the saved image colors are wrong (e.g. blue/red swapped), 
                            # we likely need to convert from RGB to BGR before saving with cv2.imwrite.
                            bgr_obs = cv2.cvtColor(obs, cv2.COLOR_RGB2BGR)
                            cv2.imwrite(img_path, bgr_obs)
                        else:
                            print(f"  Warning: Observation at {idx} has shape {obs.shape}, skipping image save.")
                            
                    except Exception as e:
                        print(f"  Error extracting index {idx} from {filename}: {e}")

        except Exception as e:
            print(f"Error opening {filename}: {e}")

    # Save collected data to new H5
    if all_observations:
        print(f"Saving {len(all_observations)} goals to {output_h5_path}")
        with h5py.File(output_h5_path, 'w') as f_out:
            f_out.create_dataset('observations', data=np.array(all_observations), compression="gzip")
            f_out.create_dataset('features', data=np.array(all_features), compression="gzip")
            f_out.create_dataset('stds', data=np.array(all_stds), compression="gzip")
            
        print("Done.")
    else:
        print("No data extracted.")

if __name__ == "__main__":
    save_evaluation_goals()
