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
            class_name = self.class_names[predicted_idx] if self.class_names and predicted_idx < len(
                self.class_names) else "Unknown"
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
    cam_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # Get camera height
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

    # --- MODIFICATION START ---
    # Create a named window that is resizable
    window_name = "Hand Gesture Recognition"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    # --- MODIFICATION END ---

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
                    # Get raw hand landmarks from MediaPipe
                    hand_landmarks = detector.get_hand_landmarks_raw()
                    display_roi_content = np.zeros((display_size, display_size, 3), dtype=np.uint8)  # Default black

                    if hand_landmarks:
                        # Create a new wireframe specifically for the ROI
                        roi_wireframe = np.zeros((roi_size, roi_size, 3), dtype=np.uint8)
                        # Get landmarks in ROI coordinates
                        h_orig, w_orig = original_img.shape[:2]

                        roi_points_for_drawing = []
                        # Convert normalized landmarks to ROI-relative pixel coordinates
                        for landmark in hand_landmarks.landmark:
                            # Pixel coordinates in original image
                            abs_x, abs_y = int(landmark.x * w_orig), int(landmark.y * h_orig)
                            # Convert to ROI-relative coordinates
                            rel_x, rel_y = abs_x - roi_x, abs_y - roi_y
                            roi_points_for_drawing.append((rel_x, rel_y))

                        # Draw connections
                        for connection in hand_connections:
                            start_idx, end_idx = connection
                            if 0 <= start_idx < len(roi_points_for_drawing) and 0 <= end_idx < len(
                                    roi_points_for_drawing):
                                p1 = roi_points_for_drawing[start_idx]
                                p2 = roi_points_for_drawing[end_idx]
                                # Draw only if points are roughly within the ROI bounds (can be slightly outside due to landmark variations)
                                # This simple check is okay as we clip to roi_wireframe size later
                                cv2.line(roi_wireframe, p1, p2, (255, 255, 255),
                                         max(1, int((roi_size / 100) * 1.5)))  # Adjusted thickness

                        # Draw landmark points
                        for point in roi_points_for_drawing:
                            cv2.circle(roi_wireframe, point, max(1, int((roi_size / 100) * 2)), (255, 255, 255),
                                       -1)  # Adjusted radius

                        # Resize the generated wireframe ROI for display
                        display_roi_content = cv2.resize(roi_wireframe, (display_size, display_size))

                        # Image for classification (based on the wireframe)
                        classification_img = cv2.resize(roi_wireframe, (128, 128))

                    else:  # No landmarks found, use grayscale ROI if available
                        gray_roi = cv2.cvtColor(hand_roi, cv2.COLOR_BGR2GRAY)
                        display_roi_content = cv2.cvtColor(cv2.resize(gray_roi, (display_size, display_size)),
                                                           cv2.COLOR_GRAY2BGR)
                        classification_img = cv2.resize(gray_roi, (128, 128))

                    # Classify the hand gesture
                    if classification_img.size > 0:
                        gesture_name, confidence, all_probs = classifier.predict(classification_img)

                    # Show the ROI in the corner of the main image
                    # Ensure roi_display area is within img bounds
                    if roi_display_y + display_size <= img.shape[0] and \
                            roi_display_x + display_size <= img.shape[1]:
                        img[roi_display_y:roi_display_y + display_size, roi_display_x:roi_display_x + display_size] \
                            = display_roi_content
                        # Add a border around the ROI display
                        cv2.rectangle(img,
                                      (roi_display_x - 2, roi_display_y - 2),
                                      (roi_display_x + display_size + 2, roi_display_y + display_size + 2),
                                      (255, 255, 255), 2)
                    else:
                        print("Warning: ROI display area is outside the main image bounds.")

                    # Save the processed image if in capture mode
                    if is_capturing and classification_img.size > 0:
                        # Create directory if it doesn't exist
                        os.makedirs(save_directory, exist_ok=True)
                        # Generate filename with timestamp
                        timestamp = int(time.time() * 1000)
                        filename = f"{save_directory}/hand_{capture_count}_{timestamp}.jpg"
                        # Save the image (ensure it's BGR if it was grayscale for classification_img)
                        if len(classification_img.shape) == 2 or classification_img.shape[2] == 1:  # Grayscale
                            save_img_for_capture = cv2.cvtColor(classification_img, cv2.COLOR_GRAY2BGR)
                        else:  # Already BGR (like roi_wireframe)
                            save_img_for_capture = classification_img
                        cv2.imwrite(filename, save_img_for_capture)
                        # Update capture count
                        capture_count += 1
                        # Display capture count on screen
                        cv2.putText(img, f"Capturing: {capture_count} images", (10, cam_height - 40),
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
            text_y_pos = roi_y + roi_size + 20 if square_roi_info else cam_height - 20
            if text_y_pos > cam_height - 10: text_y_pos = cam_height - 10  # ensure it's on screen
            text_x_pos = roi_x if square_roi_info else 10

            cv2.putText(img, f"{gesture_name} ({confidence:.1f}%)", (text_x_pos, text_y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            # Display bar chart of all class probabilities
            if all_probs is not None and classifier.class_names:
                chart_base_height = 150  # Reduced height
                chart_width = 150  # Reduced width

                # Position chart below ROI display or at bottom right if ROI display is too low
                chart_start_y = roi_display_y + display_size + 10
                if chart_start_y + chart_base_height > cam_height - 10:  # If chart goes off screen
                    chart_start_y = cam_height - chart_base_height - 10  # Move it up

                chart_start_x = roi_display_x + (display_size - chart_width) // 2  # Center under ROI display

                # Ensure chart is within bounds
                if chart_start_y < 0: chart_start_y = 10
                if chart_start_x < 0: chart_start_x = 10

                if chart_start_y + chart_base_height <= img.shape[0] and \
                        chart_start_x + chart_width <= img.shape[1]:

                    # Draw chart background
                    cv2.rectangle(img, (chart_start_x, chart_start_y),
                                  (chart_start_x + chart_width, chart_start_y + chart_base_height),
                                  (50, 50, 50), -1)
                    cv2.rectangle(img, (chart_start_x, chart_start_y),
                                  (chart_start_x + chart_width, chart_start_y + chart_base_height),
                                  (255, 255, 255), 1)  # Border

                    num_classes_to_show = len(classifier.class_names)
                    if num_classes_to_show > 0:
                        bar_spacing = 2
                        total_bar_width_area = chart_width - (num_classes_to_show + 1) * bar_spacing
                        bar_w = total_bar_width_area // num_classes_to_show if num_classes_to_show > 0 else 0

                        for i, prob in enumerate(all_probs):
                            if i >= num_classes_to_show or bar_w <= 0:
                                break

                            bar_h = int(prob * (chart_base_height - 10))  # Max height leaves space for text

                            b_x1 = chart_start_x + bar_spacing + i * (bar_w + bar_spacing)
                            b_y1 = chart_start_y + chart_base_height - bar_h - 5  # -5 for bottom margin
                            b_x2 = b_x1 + bar_w
                            b_y2 = chart_start_y + chart_base_height - 5

                            # Choose color based on probability
                            bar_color = (0, int(prob * 155) + 100, int((1 - prob) * 155))  # Gradient from red to green
                            if i == np.argmax(all_probs): bar_color = (0, 255, 0)  # Highlight best

                            cv2.rectangle(img, (b_x1, b_y1), (b_x2, b_y2), bar_color, -1)

                            # Draw class name below bar if space allows
                            class_name_short = classifier.class_names[i][:3]  # Shorten name
                            text_size, _ = cv2.getTextSize(class_name_short, cv2.FONT_HERSHEY_SIMPLEX, 0.3, 1)
                            text_x_bar = b_x1 + (bar_w - text_size[0]) // 2
                            if chart_start_y + chart_base_height + text_size[1] + 2 < img.shape[
                                0]:  # Check if text fits
                                cv2.putText(img, class_name_short,
                                            (text_x_bar, chart_start_y + chart_base_height + text_size[1] + 2),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.3, (200, 200, 200), 1)
                else:
                    print("Warning: Bar chart display area is outside the main image bounds.")


        else:
            # Display "No hand detected" message
            cv2.putText(img, "No hand detected", (10, cam_height - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        if is_capturing:
            cv2.putText(img, f"CAPTURE MODE: {save_directory.split('/')[-1]} ({capture_count})",
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)

        # Display the result
        cv2.imshow(window_name, img)
        # Check for key presses
        key = cv2.waitKey(1) & 0xFF
        # Break the loop if 'q' is pressed
        if key == ord('q'):
            break
        # Toggle capture mode if 'c' is pressed
        elif key == ord('c'):
            is_capturing = not is_capturing
            if is_capturing:
                gesture_input = input("Enter the name of the gesture to capture (or press Enter to cancel): ").strip()
                if gesture_input:  # Proceed if user entered a name
                    save_directory = f"data/custom/{gesture_input}"
                    capture_count = 0
                    print(f"Started capturing '{gesture_input}' gesture to {save_directory}")
                    print(f"Ensure directory '{save_directory}' exists or will be created.")
                else:  # User pressed enter, cancel capture mode
                    is_capturing = False
                    print("Capture mode cancelled.")

            else:
                print(f"Stopped capturing. Saved {capture_count} images.")
                # After stopping capture, refresh class names in case new folders were added
                class_names = get_class_names_from_data_folder(data_path)
                classifier.class_names = class_names  # Update classifier's class_names
                print(f"Updated classes: {', '.join(class_names if class_names else ['None'])}")
    # Release the webcam and close all windows
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()