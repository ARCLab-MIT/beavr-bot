# Test keypoint transform

import numpy as np
from beavr.components.detector.keypoint_transform import TransformHandPositionCoords
from unittest.mock import patch


# Define HAND_COORDS with the known truth values
HAND_COORDS = np.array([
    [-0.5092271, 0.712781, -0.2810469],
    [-0.5092271, 0.712781, -0.2810469],
    [-0.5248372, 0.7124779, -0.2616524],
    [-0.545522, 0.7148646, -0.2490535],
    [-0.5686282, 0.7215037, -0.2281795],
    [-0.5794767, 0.7269712, -0.1973978],
    [-0.5481126, 0.74962, -0.2001084],
    [-0.561087, 0.7492922, -0.1653092],
    [-0.5685642, 0.7360178, -0.1470252],
    [-0.5264679, 0.750635, -0.1970727],
    [-0.5388805, 0.7456619, -0.1572195],
    [-0.5472636, 0.7303123, -0.1366783],
    [-0.5082178, 0.7415956, -0.1971077],
    [-0.5151464, 0.741676, -0.1595543],
    [-0.523649, 0.7290124, -0.1384711],
    [-0.495066, 0.7155481, -0.2423497],
    [-0.491579, 0.7283962, -0.1996742],
    [-0.4848376, 0.7380876, -0.1720041],
    [-0.4836926, 0.7365084, -0.1522099],
    [-0.5903009, 0.7294336, -0.1759945],
    [-0.5734628, 0.7201086, -0.1327569],
    [-0.5546871, 0.7138456, -0.1201637],
    [-0.5334091, 0.7140357, -0.1226454],
    [-0.4850826, 0.7351671, -0.1307947]
])

@patch('beavr.components.detector.keypoint_transform.ZMQKeypointPublisher')
@patch('beavr.components.detector.keypoint_transform.ZMQKeypointSubscriber')
def test_transform_hand_position(mock_subscriber, mock_publisher):
    """Test that hand position coordinates transform correctly."""
    # Mock the methods of ZMQKeypointPublisher and ZMQKeypointSubscriber
    mock_publisher.return_value._init_publisher.return_value = None
    mock_subscriber.return_value.recv_keypoints.return_value = [1] + HAND_COORDS.flatten().tolist()

    transform = TransformHandPositionCoords(host="localhost", keypoint_port=9000, 
                                            transformation_port=9001, moving_average_limit=1)
    
    # Test with known input and expected output
    test_input = HAND_COORDS
    
    expected_output = np.array([
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0102554115, 0.0201494672, -0.0104278104],
        [0.0274948292, 0.0353299451, -0.0184680304],
        [0.0464426632, 0.0599210251, -0.0255360128],
        [0.0514861594, 0.0919955364, -0.0319301844],
        [0.0294802926, 0.092472413, -2.18987362e-18],
        [0.0339400823, 0.126834654, -0.0133706204],
        [0.0330670373, 0.140972451, -0.0324960842],
        [0.00895843722, 0.0929744212, 0.00757232982],
        [0.0104429607, 0.130621975, -0.0110700889],
        [0.00931851178, 0.146369808, -0.0329463055],
        [-0.0106770354, 0.0879281584, 0.00563285209],
        [-0.0123254299, 0.124268672, -0.00598264217],
        [-0.0126704092, 0.141362701, -0.0255999161],
        [-0.0207550684, 0.035634785, -0.00224858215],
        [-0.0294802926, 0.0794211468, 1.37911142e-18],
        [-0.0389131561, 0.107673235, 0.00422430552],
        [-0.0447245731, 0.125740624, -0.00172650356],
        [0.0576575763, 0.114294431, -0.0387125791],
        [0.0298905111, 0.150195799, -0.05213082],
        [0.00784482315, 0.157856061, -0.0545075397],
        [-0.011382814, 0.152941824, -0.046439299],
        [-0.0484558524, 0.145724444, -0.00873124943]
    ])
    
    transformed_hand_coords, _ = transform.transform_keypoints(test_input)
    assert np.allclose(transformed_hand_coords, expected_output, atol=1e-7)
