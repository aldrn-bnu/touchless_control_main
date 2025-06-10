import cv2 
import mediapipe as mp
import numpy as np
import time
from collections import deque
import pyautogui
import win32api
import win32con
from win32con import MOUSEEVENTF_WHEEL
import math

class GestureController:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_face = mp.solutions.face_detection  # Add face detection
        
        self.hands = self.mp_hands.Hands(
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7,
            max_num_hands=1
        )
        
        # Initialize face detection
        self.face_detection = self.mp_face.FaceDetection(min_detection_confidence=0.7)

        # Screen and window setup
        self.whiteboard_width = 1280
        self.whiteboard_height = 720
        
        self.screen_width, self.screen_height = pyautogui.size()
        pyautogui.FAILSAFE = False
        
        # Mouse control settings
        self.mouse_smoothing_factor = 0.5
        self.mouse_speed = 1.5
        self.last_mouse_pos = (0, 0)

        # Drawing setup
        self.whiteboard = np.ones((self.whiteboard_height, self.whiteboard_width, 3), np.uint8) * 255
        self.drawing = False
        self.draw_color = (0, 0, 0)
        self.draw_thickness = 2
        
        self.drawing_points = deque(maxlen=50)
        self.smoothing_factor = 0.15
        
        self.erasing = False
        self.eraser_size = 30
        self.eraser_color = (255, 255, 255)
        
        # State tracking
        self.prev_hand_landmarks = None
        self.prev_time = time.time()
        self.fps_history = deque(maxlen=30)
        
        # Initialize webcam
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.whiteboard_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.whiteboard_height)
        
        # Mouse state
        self.mouse_pressed = False
        self.last_click_time = 0
        self.click_cooldown = 0.3

        # Gesture states and toggles
        self.whiteboard_active = True
        self.mouse_control_active = True  # Virtual mouse toggle
        self.last_gesture_time = time.time()
        self.gesture_cooldown = 1.0
        
        # Toggle gesture state tracking
        self.toggle_wb_gesture_start = None
        self.toggle_mouse_gesture_start = None
        self.toggle_hold_duration = 1.0  # Duration to hold toggle gesture in seconds
        self.last_gesture = None
        
        # Scroll control settings
        self.scroll_active = False
        self.scroll_speed = 1.0
        self.scroll_velocity = 0
        self.smooth_scroll_amount = 0
        self.scroll_smoothing = 0.3
        self.last_scroll_y = None
        self.scroll_threshold = 0.02
        self.scroll_gesture_start_y = None
        
        # Virtual monitor settings - always active and bigger
        self.virtual_monitor_active = True  # Always enabled by default
        # Maintain the same aspect ratio (4:3) but make it bigger
        self.virtual_monitor_width = 600  # Increased from 400
        self.virtual_monitor_height = 450  # Increased from 300, keeping 4:3 ratio
        self.virtual_monitor_x = 120
        self.virtual_monitor_y = 90
        self.in_virtual_monitor = False
        self.prev_points = deque(maxlen=5)

    def get_finger_states(self, hand_landmarks):
        """Get the state of each finger (up/down) with improved accuracy"""
        finger_tips = [
            self.mp_hands.HandLandmark.THUMB_TIP,
            self.mp_hands.HandLandmark.INDEX_FINGER_TIP,
            self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
            self.mp_hands.HandLandmark.RING_FINGER_TIP,
            self.mp_hands.HandLandmark.PINKY_TIP
        ]
        
        finger_pips = [
            self.mp_hands.HandLandmark.THUMB_IP,
            self.mp_hands.HandLandmark.INDEX_FINGER_PIP,
            self.mp_hands.HandLandmark.MIDDLE_FINGER_PIP,
            self.mp_hands.HandLandmark.RING_FINGER_PIP,
            self.mp_hands.HandLandmark.PINKY_PIP
        ]

        finger_mcps = [
            self.mp_hands.HandLandmark.THUMB_MCP,
            self.mp_hands.HandLandmark.INDEX_FINGER_MCP,
            self.mp_hands.HandLandmark.MIDDLE_FINGER_MCP,
            self.mp_hands.HandLandmark.RING_FINGER_MCP,
            self.mp_hands.HandLandmark.PINKY_MCP
        ]

        # Get wrist position for reference
        wrist = hand_landmarks.landmark[self.mp_hands.HandLandmark.WRIST]

        # Check thumb separately using angle
        thumb_tip = hand_landmarks.landmark[finger_tips[0]]
        thumb_mcp = hand_landmarks.landmark[finger_mcps[0]]
        thumb_extended = (thumb_tip.x - thumb_mcp.x) > 0.05  # Adjust threshold as needed

        # Check other fingers using height comparison
        fingers_extended = [thumb_extended]
        for i in range(1, 5):  # For index through pinky
            tip = hand_landmarks.landmark[finger_tips[i]]
            pip = hand_landmarks.landmark[finger_pips[i]]
            mcp = hand_landmarks.landmark[finger_mcps[i]]
            
            # Calculate vectors
            vec_base = np.array([pip.x - mcp.x, pip.y - mcp.y, pip.z - mcp.z])
            vec_finger = np.array([tip.x - pip.x, tip.y - pip.y, tip.z - pip.z])
            
            # Normalize vectors
            vec_base = vec_base / np.linalg.norm(vec_base)
            vec_finger = vec_finger / np.linalg.norm(vec_finger)
            
            # Calculate angle between vectors
            angle = np.arccos(np.clip(np.dot(vec_base, vec_finger), -1.0, 1.0))
            
            # Consider finger extended if angle is small (finger is straight)
            fingers_extended.append(angle < 0.7)  # Threshold in radians (about 40 degrees)

        return fingers_extended

    def detect_gestures(self, hand_landmarks):
        """Enhanced gesture detection with improved reliability"""
        fingers_extended = self.get_finger_states(hand_landmarks)
        
        # Get specific landmark positions
        thumb_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.THUMB_TIP]
        index_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
        middle_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
        ring_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.RING_FINGER_TIP]
        
        # Calculate thumb-index distance for pinch detection
        thumb_index_dist = math.sqrt(
            (thumb_tip.x - index_tip.x)**2 + 
            (thumb_tip.y - index_tip.y)**2 + 
            (thumb_tip.z - index_tip.z)**2
        )

        # Define gestures with improved accuracy
        is_whiteboard_toggle = (not any(fingers_extended[:4]) and 
                              fingers_extended[4] and  # Only pinky raised
                              thumb_index_dist > 0.1)  # Ensure no pinch
                              
        # Mouse toggle gesture - pinky AND ring finger up, others down
        is_mouse_toggle = (not fingers_extended[0] and  # Thumb down
                          not fingers_extended[1] and  # Index down
                          not fingers_extended[2] and  # Middle down
                          fingers_extended[3] and      # Ring up
                          fingers_extended[4])         # Pinky up

        gestures = {
            'pinch': thumb_index_dist < 0.05,
            'eraser': fingers_extended[1] and fingers_extended[2] and not any(fingers_extended[3:]),
            'copy': fingers_extended[1] and fingers_extended[2] and fingers_extended[3] and not fingers_extended[4],
            'paste': all(fingers_extended[1:]),
            'toggle_whiteboard': is_whiteboard_toggle,
            'toggle_mouse': is_mouse_toggle
        }
        
        return gestures

    def handle_toggle_gestures(self, gestures):
        """Handle toggle gestures with proper timing"""
        current_time = time.time()
        
        # Handle whiteboard toggle
        if gestures['toggle_whiteboard']:
            if self.toggle_wb_gesture_start is None:
                self.toggle_wb_gesture_start = current_time
            elif (current_time - self.toggle_wb_gesture_start) >= self.toggle_hold_duration:
                if (current_time - self.last_gesture_time) > self.gesture_cooldown:
                    self.whiteboard_active = not self.whiteboard_active
                    self.last_gesture_time = current_time
                    self.toggle_wb_gesture_start = None
                    print(f"Whiteboard {'activated' if self.whiteboard_active else 'deactivated'}!")
        else:
            self.toggle_wb_gesture_start = None
            
        # Handle mouse toggle
        if gestures['toggle_mouse']:
            if self.toggle_mouse_gesture_start is None:
                self.toggle_mouse_gesture_start = current_time
            elif (current_time - self.toggle_mouse_gesture_start) >= self.toggle_hold_duration:
                if (current_time - self.last_gesture_time) > self.gesture_cooldown:
                    self.mouse_control_active = not self.mouse_control_active
                    self.last_gesture_time = current_time
                    self.toggle_mouse_gesture_start = None
                    print(f"Mouse control {'activated' if self.mouse_control_active else 'deactivated'}!")
        else:
            self.toggle_mouse_gesture_start = None

    def moving_average(self, new_point):
        """Smooth points using moving average"""
        self.prev_points.append(new_point)
        if len(self.prev_points) > 0:
            return np.mean(self.prev_points, axis=0, dtype=int)
        return new_point

    def handle_mouse_and_drawing(self, hand_landmarks, frame_width, frame_height, frame):
        """Enhanced handler for mouse, drawing, and gestures with improved cursor handling"""
        # Get index finger tip position
        index_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
        
        # Convert to screen coordinates
        x = int(index_tip.x * frame_width)
        y = int(index_tip.y * frame_height)
        
        # Apply mouse smoothing
        smooth_x, smooth_y = self.smooth_mouse_movement(x, y)
        
        # Check if inside virtual monitor
        self.in_virtual_monitor = (self.virtual_monitor_x < smooth_x < (self.virtual_monitor_x + self.virtual_monitor_width) and 
                                 self.virtual_monitor_y < smooth_y < (self.virtual_monitor_y + self.virtual_monitor_height))
        
        # Convert to screen coordinates - ONLY if in virtual monitor
        if self.in_virtual_monitor:
            # Map virtual monitor to full screen
            screen_x = int((smooth_x - self.virtual_monitor_x) / self.virtual_monitor_width * self.screen_width)
            screen_y = int((smooth_y - self.virtual_monitor_y) / self.virtual_monitor_height * self.screen_height)
            
            # Ensure coordinates are within screen bounds
            screen_x = max(0, min(screen_x, self.screen_width - 1))
            screen_y = max(0, min(screen_y, self.screen_height - 1))
            
            # Move mouse cursor only if mouse control is active AND pointer is inside virtual monitor
            if self.mouse_control_active:
                win32api.SetCursorPos((screen_x, screen_y))
        else:
            # If outside virtual monitor, just track position but don't move cursor
            screen_x = int((smooth_x / frame_width) * self.screen_width)
            screen_y = int((smooth_y / frame_height) * self.screen_height)
        
        # Detect gestures and handle toggles
        gestures = self.detect_gestures(hand_landmarks)
        self.handle_toggle_gestures(gestures)
        
        # Only handle scroll if mouse control is active AND in virtual monitor
        if self.mouse_control_active and self.in_virtual_monitor:
            self.handle_scroll(hand_landmarks)
        
        current_time = time.time()
        
        # Handle copy/paste gestures only if mouse control is active AND in virtual monitor
        if self.mouse_control_active and self.in_virtual_monitor and current_time - self.last_gesture_time > self.gesture_cooldown:
            if gestures['copy']:
                pyautogui.hotkey('ctrl', 'c')
                print("copied")
                self.last_gesture_time = current_time
            elif gestures['paste']:
                pyautogui.hotkey('ctrl', 'v')
                print("pasted")
                self.last_gesture_time = current_time
        
        # Handle mouse click and drawing - only if in virtual monitor
        if self.mouse_control_active and self.in_virtual_monitor and gestures['pinch'] and not self.mouse_pressed and (current_time - self.last_click_time) > self.click_cooldown:
            win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN, screen_x, screen_y, 0, 0)
            self.mouse_pressed = True
            self.last_click_time = current_time
            
            if self.whiteboard_active:
                self.drawing = True
                self.drawing_points.append((x, y))
            
        elif not gestures['pinch'] and self.mouse_pressed:
            win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP, screen_x, screen_y, 0, 0)
            self.mouse_pressed = False
            self.drawing = False
        
        # Handle drawing if whiteboard is active
        if self.whiteboard_active:
            # Only draw when in drawing mode (pinch gesture)
            if self.drawing:
                self.drawing_points.append((x, y))
                if len(self.drawing_points) >= 2:
                    smoothed_points = self.smooth_drawing(list(self.drawing_points))
                    for i in range(len(smoothed_points) - 1):
                        cv2.line(self.whiteboard, 
                                smoothed_points[i], 
                                smoothed_points[i + 1], 
                                self.draw_color, 
                                self.draw_thickness, 
                                cv2.LINE_AA)
                    self.drawing_points.clear()
                    self.drawing_points.append((x, y))
            
            if gestures['eraser']:
                cv2.circle(self.whiteboard, (x, y), self.eraser_size, self.eraser_color, -1)
        
        return x, y
    
    def handle_scroll(self, hand_landmarks):
        """Simple and reliable scroll handler"""
        fingers_extended = self.get_finger_states(hand_landmarks)
        is_scroll_gesture = (fingers_extended[1] and 
                           fingers_extended[2] and 
                           not any(fingers_extended[3:]) and 
                           not fingers_extended[0])
        
        middle_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
        current_y = middle_tip.y
        
        if is_scroll_gesture:
            if not self.scroll_active:
                self.scroll_active = True
                self.last_scroll_y = current_y
            else:
                if self.last_scroll_y is not None:
                    y_diff = current_y - self.last_scroll_y
                    if abs(y_diff) > 0.01:
                        scroll_amount = int(y_diff * 1000)
                        scroll_amount = max(min(scroll_amount, 50), -50)
                        win32api.mouse_event(MOUSEEVENTF_WHEEL, 0, 0, -scroll_amount, 0)
                    self.last_scroll_y = current_y
        else:
            self.scroll_active = False
            self.last_scroll_y = None

    def smooth_mouse_movement(self, x, y):
        """Smooth mouse movement to reduce jitter"""
        if self.last_mouse_pos == (0, 0):
            self.last_mouse_pos = (x, y)
            return x, y
            
        smoothed_x = int(x * self.mouse_smoothing_factor + self.last_mouse_pos[0] * (1 - self.mouse_smoothing_factor))
        smoothed_y = int(y * self.mouse_smoothing_factor + self.last_mouse_pos[1] * (1 - self.mouse_smoothing_factor))
        
        self.last_mouse_pos = (smoothed_x, smoothed_y)
        return smoothed_x, smoothed_y

    def smooth_drawing(self, points):
        """Smooth drawing points for better line quality"""
        if len(points) < 3:
            return points
            
        smoothed = []
        for i in range(1, len(points) - 1):
            x = int(points[i-1][0] * 0.25 + points[i][0] * 0.5 + points[i+1][0] * 0.25)
            y = int(points[i-1][1] * 0.25 + points[i][1] * 0.5 + points[i+1][1] * 0.25)
            smoothed.append((x, y))
            
        return smoothed

    def update_virtual_monitor(self, frame):
        """Update virtual monitor position based on face position"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_results = self.face_detection.process(rgb_frame)
        
        if face_results.detections:
            for detection in face_results.detections:
                bboxC = detection.location_data.relative_bounding_box
                h, w, _ = frame.shape
                head_x = int(bboxC.xmin * w + bboxC.width * w / 2)
                head_y = int(bboxC.ymin * h + bboxC.height * h / 2)
                
                # Update virtual monitor position based on head position
                self.virtual_monitor_x = max(0, head_x - self.virtual_monitor_width // 2)
                self.virtual_monitor_y = max(0, head_y - self.virtual_monitor_height // 2)
                
                # Ensure the virtual monitor stays within frame bounds
                self.virtual_monitor_x = min(self.virtual_monitor_x, w - self.virtual_monitor_width)
                self.virtual_monitor_y = min(self.virtual_monitor_y, h - self.virtual_monitor_height)
                break  # Process only the first detected face

    def process_frame(self):
        """Process each frame and update display with cursor overlay"""
        try:
            ret, frame = self.cap.read()
            if not ret:
                print("Failed to grab frame")
                return False

            frame = cv2.flip(frame, 1)
            
            # Update virtual monitor position based on face position
            self.update_virtual_monitor(frame)
            
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb_frame)

            # Create a copy of whiteboard for display
            if self.whiteboard_active:
                display = self.whiteboard.copy()
            else:
                display = frame.copy()

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    self.mp_drawing.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                    x, y = self.handle_mouse_and_drawing(hand_landmarks, frame.shape[1], frame.shape[0], frame)
                    
                    # Only draw the cursor (red dot) on the display copy, not the actual whiteboard
                    if self.whiteboard_active:
                        # Draw cursor with a more visible design
                        cv2.circle(display, (x, y), 5, (0, 0, 255), -1)  # Inner red dot
                        cv2.circle(display, (x, y), 6, (255, 255, 255), 1)  # White outline
            
            # Always draw virtual monitor rectangle since it's always active
            monitor_color = (0, 255, 0) if self.in_virtual_monitor else (255, 0, 0)
            cv2.rectangle(frame, 
                        (self.virtual_monitor_x, self.virtual_monitor_y), 
                        (self.virtual_monitor_x + self.virtual_monitor_width, self.virtual_monitor_y + self.virtual_monitor_height), 
                        monitor_color, 2)

            # Calculate and display FPS
            current_time = time.time()
            fps = 1 / (current_time - self.prev_time)
            self.prev_time = current_time
            self.fps_history.append(fps)
            avg_fps = sum(self.fps_history) / len(self.fps_history)
            
            # Display toggle gesture progress
            y_pos = 30  # Starting y position for text
            cv2.putText(frame, f'FPS: {int(avg_fps)}', (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            y_pos += 30
            
            # Status indicators
            cv2.putText(frame, f'Mouse: {"ON" if self.mouse_control_active else "OFF"}', (10, y_pos), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0) if self.mouse_control_active else (0, 0, 255), 2)
            y_pos += 30
            
            cv2.putText(frame, f'Whiteboard: {"ON" if self.whiteboard_active else "OFF"}', (10, y_pos), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0) if self.whiteboard_active else (0, 0, 255), 2)
            y_pos += 30
            
            cv2.putText(frame, f'Virtual Monitor: ON', (10, y_pos), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            y_pos += 30
            
            # Display whiteboard toggle progress if active
            if self.toggle_wb_gesture_start is not None:
                progress = min((current_time - self.toggle_wb_gesture_start) / self.toggle_hold_duration * 100, 100)
                cv2.putText(frame, f'Whiteboard Toggle: {int(progress)}%', (10, y_pos), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                y_pos += 30
                
            # Display mouse toggle progress if active
            if self.toggle_mouse_gesture_start is not None:
                progress = min((current_time - self.toggle_mouse_gesture_start) / self.toggle_hold_duration * 100, 100)
                cv2.putText(frame, f'Mouse Toggle: {int(progress)}%', (10, y_pos), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Show/hide windows based on whiteboard state
            cv2.imshow("Hand Tracking", frame)
            if self.whiteboard_active:
                cv2.imshow("Whiteboard", display)  # Show the display copy with cursor
                cv2.namedWindow("Whiteboard", cv2.WINDOW_NORMAL)
                cv2.moveWindow("Whiteboard", int(self.screen_width/4), int(self.screen_height/4))
                cv2.resizeWindow("Whiteboard", self.whiteboard_width, self.whiteboard_height)
            else:
                try:
                    cv2.destroyWindow("Whiteboard")
                except:
                    pass

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                self.running = False
            elif key == ord('c'):
                self.whiteboard = np.ones((self.whiteboard_height, self.whiteboard_width, 3), np.uint8) * 255
            elif key == ord('m'):
                self.mouse_control_active = not self.mouse_control_active
                print(f"Mouse control {'activated' if self.mouse_control_active else 'deactivated'}!")
            elif key == ord('w'):
                self.whiteboard_active = not self.whiteboard_active
                print(f"Whiteboard {'activated' if self.whiteboard_active else 'deactivated'}!")
            
            return True

        except Exception as e:
            print(f"Error in process_frame: {e}")
            return True
        
    def run(self):
        print("Starting Enhanced Gesture Controller with Always-On Virtual Monitor")
        print("\nToggle Switches:")
        print("1. Mouse Control Toggle: Ring finger UP + Pinky UP (all others down)")
        print("   - Hold for 1 second to toggle mouse control ON/OFF")
        print("   - Or press 'm' key to toggle mouse control")
        print("\n2. Whiteboard Toggle: ONLY Pinky UP (all others down)")
        print("   - Hold for 1 second to toggle whiteboard ON/OFF")
        print("   - Or press 'w' key to toggle whiteboard")
        print("\n3. Virtual Monitor: ALWAYS ON")
        print("   - Mouse control ONLY works when hand is inside virtual monitor")
        print("   - Virtual monitor tracks your face position")
        
        print("\nGesture Controls (when mouse control is ON and hand is in virtual monitor):")
        print("1. Basic Controls:")
        print("   - Move mouse: Point with index finger inside virtual monitor")
        print("   - Click/Draw: Pinch thumb and index finger")
        print("   - Erase: Hold index and middle fingers together")
        print("   - Scroll: Hold index and middle fingers up (others down)")
        print("            Move hand up/down to scroll")
        print("\n2. Advanced Controls:")
        print("   - Copy: Raise index, middle, and ring fingers (keep pinky down)")
        print("   - Paste: Raise all fingers")
        print("\n3. Keyboard Controls:")
        print("   - Clear whiteboard: Press 'c'")
        print("   - Quit program: Press 'q'")
        
        self.running = True
        try:
            while self.running:
                if not self.process_frame():
                    break
        except KeyboardInterrupt:
            print("\nGracefully shutting down...")
        except Exception as e:
            print(f"Unexpected error: {e}")
        finally:
            self.cleanup()

    def cleanup(self):
        if self.cap is not None:
            self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    controller = GestureController()
    controller.run()