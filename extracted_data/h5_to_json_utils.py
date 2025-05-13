import h5py
import json
import numpy as np
import os
from typing import Dict, Any

class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder that handles NumPy arrays and data types."""
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32,
                            np.int64, np.uint8, np.uint16, np.uint32, np.uint64)):
            return int(obj)
        if isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)

def read_h5_file(file_path: str) -> Dict[str, Any]:
    """Read an H5 file and return its contents as a dictionary."""
    data = {}
    with h5py.File(file_path, 'r') as f:
        # Recursively extract all datasets from the H5 file
        def extract_data(group, path=""):
            for key, item in group.items():
                item_path = f"{path}/{key}" if path else key
                
                if isinstance(item, h5py.Dataset):
                    # Store dataset as numpy array
                    data[item_path] = item[()]
                elif isinstance(item, h5py.Group):
                    # Recursively process group
                    extract_data(item, item_path)
                    
        extract_data(f)
    
    return data

def save_to_json(data: Dict[str, Any], output_path: str):
    """Save dictionary data to a JSON file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(data, f, cls=NumpyEncoder, indent=2)
    print(f"Data saved to {output_path}") 