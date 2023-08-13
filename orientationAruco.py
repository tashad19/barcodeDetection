import cv2
import numpy as np

# Load the ArUco dictionary (we'll use the predefined DICT_6X6_1000)
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100)


# Create the ArUco parameters
aruco_params = cv2.aruco.DetectorParameters()

# Load the camera calibration data (you may need to calibrate your camera first)
camera_matrix = np.array([[910.0, 0.0, 360.0], [0.0, 910.0, 372.0], [0.0, 0.0, 1.0]])  # Replace with your camera matrix
dist_coeffs = np.array([0.02, 0.85, 0.0003, 0.0005, -3])  # Replace with your distortion coefficients

# Initialize the video capture (replace '0' with your camera index or video file)
cap = cv2.VideoCapture(0)

while True:
    # Read a frame from the video feed
    ret, frame = cap.read()
    if not ret:
        break

    # Detect ArUco markers
    corners, ids, _ = cv2.aruco.detectMarkers(frame, aruco_dict, parameters=aruco_params)

    if ids is not None and len(ids) > 0:
        # Draw the detected markers on the frame
        cv2.aruco.drawDetectedMarkers(frame, corners, ids)

        # Estimate the pose of the first detected marker (assuming only one marker in the scene)
        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, 0.05, camera_matrix, dist_coeffs)

        print(rvecs, tvecs)

        # Draw the axis representing the orientation of the box
        for rvec, tvec in zip(rvecs, tvecs):
            cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvec, tvec, 0.05)

    # Show the frame
    cv2.imshow('Pose Estimation', frame)

    # Exit the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close all windows
cap.release()
cv2.destroyAllWindows()
