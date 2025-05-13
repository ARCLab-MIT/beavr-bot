import os
import glob
import argparse
from xarm_cartesian_states_to_json import process_xarm_cartesian_states
from joint_states_to_json import process_joint_states
from commanded_cartesian_state_to_json import process_commanded_cartesian_state
from depth_camera_to_json import process_depth_camera

def convert_demonstration_h5s_to_json(demonstration_number: int, output_dir: str = "json_data"):
    """Convert all H5 files in a demonstration folder to JSON files."""
    # Create the demonstration-specific output directory
    demo_output_dir = os.path.join(output_dir, f"demonstration_{demonstration_number}")
    os.makedirs(demo_output_dir, exist_ok=True)
    
    # Get folder with H5 files
    folder_path = f"extracted_data/demonstration_{demonstration_number}"
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"Demonstration folder {folder_path} not found")
    
    # Find all H5 files
    h5_files = glob.glob(os.path.join(folder_path, "*.h5"))
    if not h5_files:
        print(f"No H5 files found in {folder_path}")
        return []
    
    # Process each file based on its type
    results = []
    for file_path in h5_files:
        try:
            file_name = os.path.basename(file_path)
            
            if "xarm_cartesian_states" in file_name:
                output_path = process_xarm_cartesian_states(file_path, demo_output_dir)
            elif "joint_states" in file_name:
                output_path = process_joint_states(file_path, demo_output_dir)
            elif "commanded_cartesian_state" in file_name:
                output_path = process_commanded_cartesian_state(file_path, demo_output_dir)
            elif "depth" in file_name or "cam" in file_name:
                output_path = process_depth_camera(file_path, demo_output_dir)
            else:
                print(f"Unknown file type: {file_name}")
                continue
                
            results.append(output_path)
            
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
    
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert demonstration H5 files to JSON')
    parser.add_argument('--demo_number', type=int, required=True, help='Demonstration number')
    parser.add_argument('--output_dir', type=str, default="json_data", help='Output directory')
    
    args = parser.parse_args()
    
    result_files = convert_demonstration_h5s_to_json(args.demo_number, args.output_dir)
    print(f"Successfully converted {len(result_files)} files to JSON")
    for file in result_files:
        print(f"  - {file}") 