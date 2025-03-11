import cv2
import mediapipe as mp
import torch
import torch.nn.functional as f
import numpy as np
from typing import Optional, NamedTuple
import os
import time


class HandDetector:
    def __init__(self, mode=False, max_hands=2, detection_confidence=0.5, tracking_confidence=0.5):
        self.results: Optional[NamedTuple] = None
        self.mode = mode
        self.max_hands = max_hands
        self.detection_confidence = detection_confidence
        self.tracking_confidence = tracking_confidence

        # Initialize MediaPipe Hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=self.mode,
            max_num_hands=self.max_hands,
            min_detection_confidence=self.detection_confidence,
            min_tracking_confidence=self.tracking_confidence
        )
        self.mp_draw = mp.solutions.drawing_utils

        # Custom drawing specs for uniform color (red)
        self.custom_connections_style = mp.solutions.drawing_styles.DrawingSpec(
            color=(0, 0, 255), thickness=2)
        self.custom_landmark_style = mp.solutions.drawing_styles.DrawingSpec(
            color=(0, 0, 255), thickness=2, circle_radius=2)

    def find_hands(self, img, draw=True):
        # Convert BGR image to RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Process the image
        self.results = self.hands.process(img_rgb)

        # Draw hand landmarks if found and draw is True
        # Using getattr to avoid PyCharm warnings about multi_hand_landmarks
        multi_hand_landmarks = getattr(self.results, 'multi_hand_landmarks', None)
        if self.results and multi_hand_landmarks and draw:
            for hand_landmarks in multi_hand_landmarks:
                # Draw landmarks with uniform color
                self.mp_draw.draw_landmarks(
                    img,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    landmark_drawing_spec=self.custom_landmark_style,
                    connection_drawing_spec=self.custom_connections_style
                )

        return img

    def find_positions(self, img, hand_number=0):
        landmark_list = []

        # Using getattr to avoid PyCharm warnings about multi_hand_landmarks
        multi_hand_landmarks = getattr(self.results, 'multi_hand_landmarks', None)
        if self.results and multi_hand_landmarks:
            # Check if the requested hand exists
            if hand_number < len(multi_hand_landmarks):
                my_hand = multi_hand_landmarks[hand_number]

                # Get the height and width of the image
                h, w, _ = img.shape

                # Extract position of each landmark
                for hand_id, lm in enumerate(my_hand.landmark):
                    # Convert normalized coordinates to pixel coordinates
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    landmark_list.append([hand_id, cx, cy])

        return landmark_list

    def get_hand_landmarks_raw(self, hand_number=0):
        """Get the raw hand landmarks from MediaPipe"""
        multi_hand_landmarks = getattr(self.results, 'multi_hand_landmarks', None)
        if self.results and multi_hand_landmarks:
            # Check if the requested hand exists
            if hand_number < len(multi_hand_landmarks):
                return multi_hand_landmarks[hand_number]
        return None

    def get_square_hand_roi(self, img, hand_number=0, padding=20):
        """Get a square ROI that encompasses the hand with padding"""
        positions = self.find_positions(img, hand_number)
        if not positions:
            return None

        # Get min and max coordinates
        x_coords = [pos[1] for pos in positions]
        y_coords = [pos[2] for pos in positions]

        min_x = min(x_coords)
        min_y = min(y_coords)
        max_x = max(x_coords)
        max_y = max(y_coords)

        # Calculate center of the hand
        center_x = (min_x + max_x) // 2
        center_y = (min_y + max_y) // 2

        # Calculate the size needed to encompass the hand (maintain square aspect ratio)
        width = max_x - min_x + 2 * padding
        height = max_y - min_y + 2 * padding
        size = max(width, height)

        # Calculate square ROI coordinates (centered on hand)
        half_size = size // 2
        roi_x1 = max(0, center_x - half_size)
        roi_y1 = max(0, center_y - half_size)

        # Adjust if ROI goes beyond image boundaries
        if roi_x1 + size > img.shape[1]:
            roi_x1 = img.shape[1] - size
        if roi_y1 + size > img.shape[0]:
            roi_y1 = img.shape[0] - size

        # Ensure roi_x1 and roi_y1 are not negative
        roi_x1 = max(0, roi_x1)
        roi_y1 = max(0, roi_y1)

        # Ensure the square doesn't exceed image boundaries
        size = min(size, img.shape[1] - roi_x1, img.shape[0] - roi_y1)

        return roi_x1, roi_y1, size


def preprocess_image(img):
    # Convert to grayscale if it's not already
    if len(img.shape) == 3 and img.shape[2] == 3:
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        img_gray = img

    # Resize to 128x128
    img_resized = cv2.resize(img_gray, (128, 128))

    # Normalize and convert to PyTorch tensor
    img_normalized = img_resized / 255.0  # Normalize to [0, 1]
    img_normalized = (img_normalized - 0.5) / 0.5  # Normalize to [-1, 1]

    # Convert to tensor with batch dimension and channel dimension
    tensor = torch.from_numpy(img_normalized).float().unsqueeze(0).unsqueeze(0)

    return tensor


class GestureClassifier:
    def __init__(self, model_path, class_names=None):
        # Load the PyTorch model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = torch.jit.load(model_path, map_location=self.device)
        self.model.eval()  # Set model to evaluation mode

        # Define class names
        self.class_names = class_names

    def predict(self, img):
        # Preprocess the image
        tensor = preprocess_image(img)

        # Move to device
        tensor = tensor.to(self.device)

        # Get prediction
        with torch.no_grad():
            outputs = self.model(tensor)
            probabilities = f.softmax(outputs, dim=1)[0]

            # Get top prediction
            confidence, predicted_class = torch.max(probabilities, 0)

            # Convert to Python values
            predicted_idx = predicted_class.item()
            confidence_score = confidence.item() * 100  # Convert to percentage

            # Get class name
            class_name = self.class_names[predicted_idx] if predicted_idx < len(self.class_names) else "Unknown"

            return class_name, confidence_score, probabilities.cpu().numpy()


def get_class_names_from_data_folder(data_path="data/custom"):
    """Automatically discover class names from subdirectories in the data folder"""
    try:
        # Get all subdirectories in the data folder
        class_names = [d for d in os.listdir(data_path)
                       if os.path.isdir(os.path.join(data_path, d)) and not d.startswith('.')]
        # Sort alphabetically for consistency
        class_names.sort()
    except Exception as e:
        print(f"Error reading class names from {data_path}: {e}")
        # Fallback to default class names
        class_names = ["love", "peace", "tmb_down", "tmb_up"]

    return class_names


def draw_wireframe_on_black(img_shape, landmarks, connections, line_thickness=2, point_radius=3):
    """
    Draw a white wireframe on black background based on MediaPipe landmarks

    Args:
        img_shape: Shape of the image (height, width, channels)
        landmarks: MediaPipe hand landmarks
        connections: MediaPipe hand connections
        line_thickness: Thickness of the wireframe lines
        point_radius: Radius of the landmark points

    Returns:
        White wireframe on black background image
    """
    # Create a black canvas
    height, width = img_shape[:2]
    wireframe_img = np.zeros((height, width, 3), dtype=np.uint8)

    # Extract landmark points
    points = []
    for landmark in landmarks.landmark:
        x, y = int(landmark.x * width), int(landmark.y * height)
        points.append((x, y))

    # Draw the connections as white lines
    for connection in connections:
        start_idx = connection[0]
        end_idx = connection[1]

        if start_idx < len(points) and end_idx < len(points):
            start_point = points[start_idx]
            end_point = points[end_idx]
            cv2.line(wireframe_img, start_point, end_point, (255, 255, 255), line_thickness)

    # Draw the landmarks as white dots
    for point in points:
        cv2.circle(wireframe_img, point, point_radius, (255, 255, 255), -1)

    return wireframe_img


def main():
    # Discover class names from data folder
    data_path = "data/custom"
    class_names = get_class_names_from_data_folder(data_path)

    print(f"Discovered {len(class_names)} classes: {', '.join(class_names)}")

    # Check if model file exists
    model_path = "best_hand_gesture_classifier.pt"
    if not os.path.exists(model_path):
        print(f"Error: Model file '{model_path}' not found.")
        print("Please make sure the trained model is in the current directory.")
        return

    # Initialize classifier
    classifier = GestureClassifier(model_path, class_names)

    # Initialize webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return

    # Get camera dimensions
    cam_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    # Initialize hand detector
    detector = HandDetector(max_hands=1)

    # Fixed size for ROI display
    display_size = 200

    # Fixed ROI display position in main window
    roi_display_x = cam_width - display_size - 20  # 20 pixels from right
    roi_display_y = 20  # 20 pixels from top

    # Initialize capture mode variables
    is_capturing = False
    capture_count = 0
    save_directory = "data/custom/new_gesture"  # Default directory for new gestures

    # Get MediaPipe hand connections for drawing
    mp_hands = mp.solutions.hands
    hand_connections = [(connection[0], connection[1]) for connection in mp_hands.HAND_CONNECTIONS]

    print("Starting hand gesture recognition...")
    print("Controls:")
    print("  - Press 'c' to toggle capture mode for collecting more training data")
    print("  - Press 'q' to quit")

    while True:
        # Read frame from webcam
        success, img = cap.read()

        if not success:
            print("Failed to grab frame")
            break

        # Make a copy of the original image before adding wireframes
        original_img = img.copy()

        # Find hands in the frame and draw wireframes on the main display
        img = detector.find_hands(img, draw=True)

        # Process the ROI without wireframes using the original image
        square_roi_info = detector.get_square_hand_roi(original_img, padding=50)
        roi_x = 0
        roi_y = 0
        roi_size = 0

        # Variable for classification result
        gesture_name = "No hand detected"
        confidence = 0.0
        all_probs = None

        if square_roi_info:
            roi_x, roi_y, roi_size = square_roi_info

            # Draw the square ROI on the main display
            cv2.rectangle(img, (roi_x, roi_y), (roi_x + roi_size, roi_y + roi_size),
                          (0, 255, 0), 2)

            # Extract ROI if it has valid dimensions
            if roi_size > 0 and roi_x + roi_size <= original_img.shape[1] and roi_y + roi_size <= original_img.shape[0]:
                # Extract the hand region from original image (no wireframe)
                hand_roi = original_img[roi_y:roi_y + roi_size, roi_x:roi_x + roi_size].copy()

                if hand_roi.size > 0:  # Check if ROI is not empty
                    # Convert to grayscale for classification
                    gray_roi = cv2.cvtColor(hand_roi, cv2.COLOR_BGR2GRAY)

                    # Get raw hand landmarks from MediaPipe
                    hand_landmarks = detector.get_hand_landmarks_raw()

                    # Process display based on hand landmarks
                    if hand_landmarks:
                        # Create a black canvas with white wireframe
                        wireframe_img = draw_wireframe_on_black(
                            hand_roi.shape, hand_landmarks, hand_connections,
                            line_thickness=2, point_radius=4
                        )

                        # Extract the ROI area from the wireframe image
                        # (Need to translate the landmarks to ROI coordinates)
                        h, w = original_img.shape[:2]

                        # Create a new wireframe specifically for the ROI
                        roi_wireframe = np.zeros((roi_size, roi_size, 3), dtype=np.uint8)

                        # Get landmarks in ROI coordinates
                        roi_points = []
                        for landmark in hand_landmarks.landmark:
                            # Convert normalized coordinates to pixel coordinates in original image
                            x, y = int(landmark.x * w), int(landmark.y * h)

                            # Convert to ROI-relative coordinates
                            roi_x_coord = x - roi_x
                            roi_y_coord = y - roi_y

                            # Only include points that are within the ROI
                            if 0 <= roi_x_coord < roi_size and 0 <= roi_y_coord < roi_size:
                                roi_points.append((roi_x_coord, roi_y_coord))

                        # Draw the connections between landmarks
                        for connection in hand_connections:
                            start_idx = connection[0]
                            end_idx = connection[1]

                            if (start_idx < len(hand_landmarks.landmark) and
                                    end_idx < len(hand_landmarks.landmark)):
                                # Calculate start point in ROI coordinates
                                start_x = int(hand_landmarks.landmark[start_idx].x * w) - roi_x
                                start_y = int(hand_landmarks.landmark[start_idx].y * h) - roi_y

                                # Calculate end point in ROI coordinates
                                end_x = int(hand_landmarks.landmark[end_idx].x * w) - roi_x
                                end_y = int(hand_landmarks.landmark[end_idx].y * h) - roi_y

                                # Only draw if both points are within the ROI
                                if (0 <= start_x < roi_size and 0 <= start_y < roi_size and
                                        0 <= end_x < roi_size and 0 <= end_y < roi_size):
                                    cv2.line(roi_wireframe,
                                             (start_x, start_y),
                                             (end_x, end_y),
                                             (255, 255, 255), int((roi_size/100)*3))

                        # Draw the landmark points
                        for point in roi_points:
                            cv2.circle(roi_wireframe, point, int((roi_size/100)*3), (255, 255, 255), -1)

                        # Resize to display size
                        display_roi = cv2.resize(roi_wireframe, (display_size, display_size))
                    else:
                        # Fallback - just create a black image for display
                        display_roi = np.zeros((display_size, display_size, 3), dtype=np.uint8)

                    # Resize to 128x128
                    hand_img = cv2.resize(display_roi, (128, 128))

                    # Classify the hand gesture using the original grayscale ROI
                    gesture_name, confidence, all_probs = classifier.predict(hand_img)

                    # Show the ROI in the corner of the main image
                    img[roi_display_y:roi_display_y + display_size, roi_display_x:roi_display_x + display_size]\
                        = display_roi

                    # Add a border around the ROI display
                    cv2.rectangle(img,
                                  (roi_display_x - 2, roi_display_y - 2),
                                  (roi_display_x + display_size + 2, roi_display_y + display_size + 2),
                                  (255, 255, 255), 2)

                    # Save the processed image if in capture mode
                    if is_capturing:
                        # Create directory if it doesn't exist
                        os.makedirs(save_directory, exist_ok=True)

                        # Generate filename with timestamp
                        timestamp = int(time.time() * 1000)
                        filename = f"{save_directory}/hand_{capture_count}_{timestamp}.jpg"

                        # Save the image
                        cv2.imwrite(filename, hand_img)

                        # Update capture count
                        capture_count += 1

                        # Display capture count on screen
                        cv2.putText(img, f"Capturing: {capture_count} images", (10, 15),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        # Display classification results
        if gesture_name != "No hand detected":
            # Create colored text based on confidence
            if confidence > 90:
                color = (0, 255, 0)  # Green for high confidence
            elif confidence > 70:
                color = (0, 255, 255)  # Yellow for medium confidence
            else:
                color = (0, 0, 255)  # Red for low confidence

            # Display the gesture name and confidence
            cv2.putText(img, f"Gesture: {gesture_name} {confidence:.2f}%", (roi_x, roi_y + roi_size + 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Display bar chart of all class probabilities
            if 'all_probs' in locals():
                chart_height = 200
                chart_width = 200
                if len(class_names) != 0:
                    bar_width = chart_width // len(class_names)
                else:
                    bar_width = chart_width
                chart_x = roi_display_x
                chart_y = roi_display_y + chart_height + 20

                # Draw chart background
                cv2.rectangle(img, (chart_x, chart_y),
                              (chart_x + chart_width, chart_y + chart_height),
                              (50, 50, 50), -1)

                # Draw bars for each class
                for i, prob in enumerate(all_probs):
                    if i >= len(class_names):
                        break

                    # Calculate bar height (percentage of chart height)
                    bar_height = int(prob * chart_height)

                    # Calculate bar position
                    x1 = chart_x + i * bar_width
                    y1 = chart_y + chart_height - bar_height
                    x2 = x1 + bar_width - 5  # gap between bars
                    y2 = chart_y + chart_height

                    # Choose color based on probability
                    if prob > 0.7:
                        bar_color = (0, 255, 0)  # Green
                    elif prob > 0.3:
                        bar_color = (0, 255, 255)  # Yellow
                    else:
                        bar_color = (0, 128, 255)  # Orange

                    # Add a border around the bar chart
                    cv2.rectangle(img,
                                  (chart_x, chart_y),
                                  (chart_x + chart_width, chart_y + chart_height),
                                  (255, 255, 255),
                                  2)

                    # Draw the bar
                    cv2.rectangle(img, (x1, y1), (x2, y2), bar_color, -1)

        else:
            # Display "No hand detected" message
            cv2.putText(img, "No hand detected", (10, 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        # Display the result
        cv2.imshow("Hand Gesture Recognition", img)

        # Check for key presses
        key = cv2.waitKey(1) & 0xFF

        # Break the loop if 'q' is pressed
        if key == ord('q'):
            break

        # Toggle capture mode if 'c' is pressed
        elif key == ord('c'):
            is_capturing = not is_capturing
            if is_capturing:
                gesture_name = input("Enter the name of the gesture to capture: ")
                save_directory = f"data/custom/{gesture_name}"
                capture_count = 0
                print(f"Started capturing '{gesture_name}' gesture to {save_directory}")
            else:
                print(f"Stopped capturing. Saved {capture_count} images.")
                # After stopping capture, refresh class names in case new folders were added
                class_names = get_class_names_from_data_folder(data_path)
                print(f"Updated classes: {', '.join(class_names)}")

    # Release the webcam and close all windows
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
