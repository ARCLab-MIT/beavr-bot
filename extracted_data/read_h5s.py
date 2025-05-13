import h5py
import os
import glob
from typing import Dict, Any, List
from dataclasses import dataclass

DEMOSTRATION_NUMBER = 271 # Change this number as needed

@dataclass
class DemonstrationData:
    """Class to store data from a demonstration H5 file."""
    file_path: str
    data: Dict[str, Any]

def read_demonstration_h5s(demonstration_number: int) -> List[DemonstrationData]:
    """
    Read H5 files from the demonstration folder with the specified number.
    
    Args:
        demonstration_number: The number of the demonstration folder to read from
    
    Returns:
        A list of DemonstrationData objects containing the H5 file contents
    """
    folder_path = f"extracted_data/demonstration_{demonstration_number}"
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"Demonstration folder {folder_path} not found")
    
    h5_files = glob.glob(os.path.join(folder_path, "*.h5"))
    if not h5_files:
        print(f"No H5 files found in {folder_path}")
        return []
    
    results = []
    for file_path in h5_files:
        try:
            data = read_h5_file(file_path)
            results.append(data)
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
    
    return results

def read_h5_file(file_path: str) -> DemonstrationData:
    """
    Read an individual H5 file and extract its data.
    
    Args:
        file_path: Path to the H5 file
        
    Returns:
        DemonstrationData object containing the file's contents
    """
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
        
    return DemonstrationData(file_path=file_path, data=data)

def main():
    # Set the demonstration number here - no user input required
    demonstration_number = DEMOSTRATION_NUMBER
    
    try:
        data_list = read_demonstration_h5s(demonstration_number)
        print(f"Successfully read {len(data_list)} H5 files from demonstration_{demonstration_number}")
        
        # Process the data as needed
        for i, data in enumerate(data_list):
            print(f"File {i+1}: {os.path.basename(data.file_path)}")
            print(f"  Data keys: {list(data.data.keys())[:5]}...")
            print(f"  Total datasets: {len(data.data)}")
            print("-" * 50)
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
