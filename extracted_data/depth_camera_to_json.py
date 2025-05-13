import os
import argparse
from h5_to_json_utils import read_h5_file, save_to_json

def process_depth_camera(h5_path: str, output_dir: str = "json_data"):
    """Process depth camera H5 file and save as JSON."""
    # Read H5 file
    data = read_h5_file(h5_path)
    
    # Create structured JSON data
    json_data = {
        "metadata": {
            "file_name": data.get("file_name", "").decode() if isinstance(data.get("file_name", ""), bytes) else data.get("file_name", ""),
            "num_datapoints": int(data.get("num_datapoints", 0)),
            "record_duration": float(data.get("record_duration", 0)),
            "record_end_time": float(data.get("record_end_time", 0))
        }
    }
    
    # Note: Depth images can be large, so we'll reference them separately
    # Instead of including them directly in the JSON
    depth_images_output = os.path.join(output_dir, "depth_images_data.npy")
    if "depth_images" in data:
        import numpy as np
        os.makedirs(output_dir, exist_ok=True)
        np.save(depth_images_output, data["depth_images"])
        json_data["depth_images_path"] = depth_images_output
    
    # Create output filename
    base_name = os.path.basename(h5_path).replace(".h5", ".json")
    output_path = os.path.join(output_dir, base_name)
    
    # Save to JSON
    save_to_json(json_data, output_path)
    
    return output_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert depth camera H5 to JSON')
    parser.add_argument('--h5_path', type=str, required=True, help='Path to H5 file')
    parser.add_argument('--output_dir', type=str, default="json_data", help='Output directory')
    
    args = parser.parse_args()
    process_depth_camera(args.h5_path, args.output_dir) 