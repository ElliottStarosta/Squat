import cv2
import mediapipe as mp
import numpy as np
import math
import time
import json
import requests
from datetime import datetime

class SquatAnalyzer:
    def __init__(self, api_key=None):
        # Initialize MediaPipe Pose
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,  # Use medium complexity for balance of speed/accuracy
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Initialize ChatGPT API key
        self.api_key = api_key
        
        # Initialize variables for squat phase detection
        self.current_phase = "standing"
        self.prev_knee_angle = 180
        
        # For angle history and smoothing
        self.knee_angle_history = []
        self.hip_angle_history = []
        self.back_angle_history = []
        
        # For squat rep tracking
        self.squat_count = 0
        self.in_squat = False
        self.bottom_reached = False
        
        # Store squat data for analysis
        self.squat_data = []
        self.current_squat = {}
        
        # For phase stability
        self.phase_stability_counter = 0
        self.phase_stability_threshold = 3
        self.tentative_phase = None
        
        # For feedback frequency control
        self.last_feedback_time = time.time()
        self.feedback_cooldown = 10  # seconds between feedback
        self.current_feedback = ""
    
    def calculate_angle(self, a, b, c):
        """Calculate the angle between three points"""
        a = np.array(a)
        b = np.array(b)
        c = np.array(c)
        
        radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
        angle = np.abs(radians * 180.0 / np.pi)
        
        if angle > 180.0:
            angle = 360 - angle
            
        return angle
    
    def detect_pose(self, frame):
        """Detect pose landmarks using MediaPipe"""
        # Convert the BGR image to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame and get the pose landmarks
        results = self.pose.process(rgb_frame)
        
        return results
    
    def draw_pose(self, frame, results):
        """Draw the pose landmarks and connections on the frame"""
        if results.pose_landmarks:
            self.mp_drawing.draw_landmarks(
                frame, 
                results.pose_landmarks, 
                self.mp_pose.POSE_CONNECTIONS,
                self.mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                self.mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
            )
        
        return frame
    
    def extract_features(self, results, frame_height, frame_width):
        """Extract relevant features from the pose landmarks"""
        if not results.pose_landmarks:
            return None
        
        landmarks = results.pose_landmarks.landmark
        
        # Get coordinates of key points
        # Hip, knee, and ankle points
        left_hip = [landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value].x * frame_width,
                    landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value].y * frame_height]
        left_knee = [landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE.value].x * frame_width,
                     landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE.value].y * frame_height]
        left_ankle = [landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE.value].x * frame_width,
                      landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE.value].y * frame_height]
        left_shoulder = [landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].x * frame_width,
                         landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].y * frame_height]
        
        right_hip = [landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value].x * frame_width,
                     landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value].y * frame_height]
        right_knee = [landmarks[self.mp_pose.PoseLandmark.RIGHT_KNEE.value].x * frame_width,
                      landmarks[self.mp_pose.PoseLandmark.RIGHT_KNEE.value].y * frame_height]
        right_ankle = [landmarks[self.mp_pose.PoseLandmark.RIGHT_ANKLE.value].x * frame_width,
                       landmarks[self.mp_pose.PoseLandmark.RIGHT_ANKLE.value].y * frame_height]
        right_shoulder = [landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x * frame_width,
                          landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y * frame_height]
        
        # Calculate key angles
        left_knee_angle = self.calculate_angle(left_hip, left_knee, left_ankle)
        right_knee_angle = self.calculate_angle(right_hip, right_knee, right_ankle)
        knee_angle = (left_knee_angle + right_knee_angle) / 2  # Average both knees
        
        left_hip_angle = self.calculate_angle(left_shoulder, left_hip, left_knee)
        right_hip_angle = self.calculate_angle(right_shoulder, right_hip, right_knee)
        hip_angle = (left_hip_angle + right_hip_angle) / 2  # Average both hips
        
        # Calculate back angle (relative to vertical)
        back_x = (left_shoulder[0] + right_shoulder[0]) / 2 - (left_hip[0] + right_hip[0]) / 2
        back_y = (left_shoulder[1] + right_shoulder[1]) / 2 - (left_hip[1] + right_hip[1]) / 2
        vertical_angle = abs(math.degrees(math.atan2(back_x, -back_y)))  # Angle with vertical
        
        # Calculate knee position relative to ankle (to evaluate knee tracking)
        left_knee_over_ankle = (left_knee[0] - left_ankle[0]) / frame_width
        right_knee_over_ankle = (right_knee[0] - right_ankle[0]) / frame_width
        knee_alignment = (left_knee_over_ankle + right_knee_over_ankle) / 2
        
        # Smooth the angles using a moving average
        self.knee_angle_history.append(knee_angle)
        self.hip_angle_history.append(hip_angle)
        self.back_angle_history.append(vertical_angle)
        
        if len(self.knee_angle_history) > 5:
            self.knee_angle_history.pop(0)
            self.hip_angle_history.pop(0)
            self.back_angle_history.pop(0)
        
        smoothed_knee_angle = sum(self.knee_angle_history) / len(self.knee_angle_history)
        smoothed_hip_angle = sum(self.hip_angle_history) / len(self.hip_angle_history)
        smoothed_back_angle = sum(self.back_angle_history) / len(self.back_angle_history)
        
        # Return all calculated features
        features = {
            'knee_angle': smoothed_knee_angle,
            'hip_angle': smoothed_hip_angle,
            'back_angle': smoothed_back_angle,
            'knee_alignment': knee_alignment,
        }
        
        return features
    
    def detect_squat_phase(self, knee_angle):
        """Detect the current phase of the squat with improved stability"""
        # Thresholds for squat phases
        BOTTOM_THRESHOLD = 110  # More lenient threshold for bottom detection
        TOP_THRESHOLD = 160     # More flexible top threshold
        
        if len(self.knee_angle_history) >= 2:
            angle_change = knee_angle - self.prev_knee_angle
            
            # Determine tentative phase
            if knee_angle >= TOP_THRESHOLD:
                new_tentative_phase = "standing"
            elif knee_angle <= BOTTOM_THRESHOLD:
                new_tentative_phase = "bottom"
            elif angle_change < -1:  # Moving down
                new_tentative_phase = "descending"
            elif angle_change > 1:   # Moving up
                new_tentative_phase = "ascending"
            else:
                new_tentative_phase = self.current_phase  # Maintain current if small change
            
            # Phase stability mechanism
            if new_tentative_phase != self.tentative_phase:
                self.tentative_phase = new_tentative_phase
                self.phase_stability_counter = 0
            else:
                self.phase_stability_counter += 1
            
            # Only change phase after stabilizing for several frames
            if self.phase_stability_counter >= self.phase_stability_threshold:
                # Track squat reps
                if self.tentative_phase == "bottom" and self.current_phase != "bottom":
                    self.bottom_reached = True
                    # Record bottom position data
                    self.current_squat["bottom"] = {
                        'knee_angle': knee_angle,
                        'hip_angle': self.hip_angle_history[-1] if self.hip_angle_history else 0,
                        'back_angle': self.back_angle_history[-1] if self.back_angle_history else 0,
                    }
                elif self.tentative_phase == "standing" and self.current_phase != "standing" and self.bottom_reached:
                    self.squat_count += 1
                    self.bottom_reached = False
                    # Record top position data and complete the squat record
                    self.current_squat["top"] = {
                        'knee_angle': knee_angle,
                        'hip_angle': self.hip_angle_history[-1] if self.hip_angle_history else 0,
                        'back_angle': self.back_angle_history[-1] if self.back_angle_history else 0,
                    }
                    self.current_squat["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    self.squat_data.append(self.current_squat.copy())
                    self.current_squat = {}
                    
                    # Generate feedback after completing a squat
                    if time.time() - self.last_feedback_time > self.feedback_cooldown:
                        self.generate_feedback()
                
                # Update current phase
                self.current_phase = self.tentative_phase
        
        self.prev_knee_angle = knee_angle
        return self.current_phase
    
    def generate_feedback(self):
        """Generate feedback using ChatGPT API based on recent squat data"""
        if not self.api_key or not self.squat_data:
            return
        
        # Get the most recent completed squat
        last_squat = self.squat_data[-1]
        
        # Prepare data for ChatGPT
        squat_metrics = {
            "bottom_knee_angle": last_squat.get("bottom", {}).get("knee_angle", 0),
            "bottom_hip_angle": last_squat.get("bottom", {}).get("hip_angle", 0),
            "bottom_back_angle": last_squat.get("bottom", {}).get("back_angle", 0),
            "top_knee_angle": last_squat.get("top", {}).get("knee_angle", 0),
            "top_hip_angle": last_squat.get("top", {}).get("hip_angle", 0),
            "top_back_angle": last_squat.get("top", {}).get("back_angle", 0),
        }
        
        # Create prompt for ChatGPT
        prompt = f"""
        As a professional fitness coach, analyze this squat form data and provide specific feedback:
        
        Knee angle at bottom: {squat_metrics['bottom_knee_angle']:.1f}° (Ideal: 90-110°)
        Hip angle at bottom: {squat_metrics['bottom_hip_angle']:.1f}° (Ideal: 50-70°)
        Back angle at bottom: {squat_metrics['bottom_back_angle']:.1f}° (Ideal: 0-30°, vertical is 0°)
        
        Provide 2-3 specific form corrections or strengths in a concise bullet point format.
        Highlight the most important area to improve first.
        """
        
        try:
            # Call ChatGPT API
            response = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "gpt-3.5-turbo",
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 150,
                    "temperature": 0.7
                }
            )
            
            # Process the response
            if response.status_code == 200:
                feedback = response.json()["choices"][0]["message"]["content"].strip()
                self.current_feedback = feedback
                self.last_feedback_time = time.time()
            else:
                print(f"API Error: {response.status_code}")
                self.current_feedback = "API error: Could not generate feedback"
        except Exception as e:
            print(f"Error generating feedback: {e}")
            self.current_feedback = "Error generating feedback"
    
    def rule_based_feedback(self, features):
        """Generate simple rule-based feedback when API is not available"""
        feedback = []
        
        # Check knee angle for depth
        if features['knee_angle'] < 80:
            feedback.append("• Going too deep - could stress knee joints")
        elif features['knee_angle'] > 120:
            feedback.append("• Not squatting deep enough - aim for deeper bend")
        
        # Check back angle
        if features['back_angle'] > 45:
            feedback.append("• Keep your back more upright (too much forward lean)")
        
        # Check knee position
        if features['knee_alignment'] < 0:
            feedback.append("• Knees should track forward slightly over toes")
        elif features['knee_alignment'] > 0.15:
            feedback.append("• Knees tracking too far forward - weight on heels")
        
        # Check hip hinge
        if features['hip_angle'] > 120:
            feedback.append("• Hinge more at the hips when descending")
        
        if not feedback:
            feedback.append("• Good form! Nice work on this rep.")
        
        return feedback

def main():
    # Initialize the squat analyzer - provide your OpenAI API key if you want ChatGPT feedback
    # If no API key provided, will use rule-based feedback instead
    analyzer = SquatAnalyzer(api_key=None)  # Replace with "your_api_key_here" BUT USE AN ENV DONT PUT THE API KEY IN THE FUCKING CODE MATHEW!
    
    # Open camera
    cap = cv2.VideoCapture(0)
    
    # For displaying text with background
    def put_text_with_background(img, text, position, font=cv2.FONT_HERSHEY_SIMPLEX, 
                               font_scale=0.7, color=(255, 255, 255), thickness=1, 
                               bg_color=(0, 0, 0), padding=5):
        (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)
        cv2.rectangle(img, 
                    (position[0] - padding, position[1] - text_height - padding), 
                    (position[0] + text_width + padding, position[1] + padding), 
                    bg_color, -1)
        cv2.putText(img, text, position, font, font_scale, color, thickness)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # Flip the frame horizontally for a selfie-view
        frame = cv2.flip(frame, 1)
        
        # Get frame dimensions
        frame_height, frame_width, _ = frame.shape
        
        # Detect pose
        pose_results = analyzer.detect_pose(frame)
        
        # Draw pose on frame
        frame = analyzer.draw_pose(frame, pose_results)
        
        # Extract features
        features = analyzer.extract_features(pose_results, frame_height, frame_width)
        
        # Process squat detection and analysis
        if features:
            # Detect squat phase
            current_phase = analyzer.detect_squat_phase(features['knee_angle'])
            
            # If no ChatGPT API key provided, use rule-based feedback
            if analyzer.api_key is None and current_phase == "standing" and analyzer.squat_count > 0:
                if time.time() - analyzer.last_feedback_time > analyzer.feedback_cooldown:
                    analyzer.current_feedback = "\n".join(analyzer.rule_based_feedback(features))
                    analyzer.last_feedback_time = time.time()
            
            # Display current metrics with background
            put_text_with_background(frame, f"Phase: {current_phase.upper()}", (10, 30))
            put_text_with_background(frame, f"Knee angle: {features['knee_angle']:.1f}°", (10, 60))
            put_text_with_background(frame, f"Hip angle: {features['hip_angle']:.1f}°", (10, 90))
            put_text_with_background(frame, f"Back angle: {features['back_angle']:.1f}°", (10, 120))
            
            # Display squat count
            put_text_with_background(frame, f"Squats completed: {analyzer.squat_count}", 
                                   (200, 30), bg_color=(0, 100, 0))
            
            # Create color coding for phase
            phase_colors = {
                "standing": (0, 255, 0),    # Green
                "descending": (0, 165, 255), # Orange
                "bottom": (0, 0, 255),      # Red
                "ascending": (255, 0, 0)    # Blue
            }
            
            # Display phase indicator
            cv2.rectangle(frame, 
                        (frame_width - 40, 60), 
                        (frame_width - 10, 90), 
                        phase_colors.get(current_phase, (200, 200, 200)), -1)
            
            # Display feedback if available
            if analyzer.current_feedback:
                feedback_lines = analyzer.current_feedback.strip().split('\n')
                y_offset = frame_height - 120
                
                # Add semi-transparent overlay for feedback
                overlay = frame.copy()
                cv2.rectangle(overlay, (0, y_offset - 30), (frame_width, frame_height), 
                            (0, 0, 0), -1)
                cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
                
                # Add heading
                put_text_with_background(frame, "FORM FEEDBACK:", 
                                       (10, y_offset - 10), 
                                       font_scale=0.8, color=(255, 255, 255), 
                                       bg_color=(0, 100, 200), padding=5)
                
                # Add feedback lines
                for i, line in enumerate(feedback_lines):
                    cv2.putText(frame, line, (10, y_offset + i * 25 + 15), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        
        cv2.namedWindow('Squat Form Analyzer', cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty('Squat Form Analyzer', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        # Display the frame
        cv2.imshow('Squat Form Analyzer', frame)
        
        # Exit on 'q' press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    
    # Save squat data for later analysis
    if analyzer.squat_data:
        with open('squat_session_data.json', 'w') as f:
            json.dump(analyzer.squat_data, f)
        print(f"Saved data for {len(analyzer.squat_data)} squats to squat_session_data.json")

if __name__ == "__main__":
    main()
