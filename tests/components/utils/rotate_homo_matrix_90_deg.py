from scipy.spatial.transform.rotation import Rotation


# Create a rotation matrix for 90 degrees around x-axis directly
R = Rotation.from_euler('x', 90, degrees=True).as_matrix()

print(R)
