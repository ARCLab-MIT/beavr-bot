import os
import argparse
from h5_to_json_utils import read_h5_file, save_to_json

def process_commanded_cartesian_state(h5_path: str, output_dir: str = "json_data"):
    """Process commanded cartesian state H5 file and save as JSON."""
    # Read H5 file
    data = read_h5_file(h5_path)
    
    # Create structured JSON data
    json_data = {
        "global_timestamps": data.get("timestamps", []).tolist(),
        "commanded_cartesian_state": {
            "position": data.get("commanded_cartesian_state/commanded_cartesian_position", []).tolist(),
            "timestamps": data.get("commanded_cartesian_state/timestamps", []).tolist()
        }
    }
    
    # Create output filename
    base_name = os.path.basename(h5_path).replace(".h5", ".json")
    output_path = os.path.join(output_dir, base_name)
    
    # Save to JSON
    save_to_json(json_data, output_path)
    
    return output_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert commanded cartesian state H5 to JSON')
    parser.add_argument('--h5_path', type=str, required=True, help='Path to H5 file')
    parser.add_argument('--output_dir', type=str, default="json_data", help='Output directory')
    
    args = parser.parse_args()
    process_commanded_cartesian_state(args.h5_path, args.output_dir) 