import os
import argparse
from h5_to_json_utils import read_h5_file, save_to_json

def process_xarm_cartesian_states(h5_path: str, output_dir: str = "json_data"):
    """Process xarm cartesian states H5 file and save as JSON."""
    # Read H5 file
    data = read_h5_file(h5_path)
    
    # Create structured JSON data
    json_data = {
        "global_timestamps": data.get("timestamps", []).tolist(),
        "cartesian_position": {
            "data": data.get("xarm_cartesian_states/cartesian_position", []).tolist(),
            "timestamps": data.get("xarm_cartesian_states/timestamps", []).tolist()
        }
    }
    
    # Create output filename
    base_name = os.path.basename(h5_path).replace(".h5", ".json")
    output_path = os.path.join(output_dir, base_name)
    
    # Save to JSON
    save_to_json(json_data, output_path)
    
    return output_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert XArm cartesian states H5 to JSON')
    parser.add_argument('--h5_path', type=str, required=True, help='Path to H5 file')
    parser.add_argument('--output_dir', type=str, default="json_data", help='Output directory')
    
    args = parser.parse_args()
    process_xarm_cartesian_states(args.h5_path, args.output_dir) 