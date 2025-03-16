import cv2
import numpy as np
import time
import threading
from PIL import Image, ImageDraw

class IntrusionDetector:
    def __init__(self):
        # Initialize the face detector using OpenCV's built-in Haar cascade
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Initialize variables
        self.reference_face = None
        self.face_bbox = None
        self.monitoring = False
        self.intrusion_detected = False
        self.intrusion_count = 0
        self.last_intrusion_time = 0
        self.monitor_thread = None
        self.camera = None
        self.frame_width = 640
        self.frame_height = 480
        self.bbox_padding = 20  # Padding around the face for the bounding box
        
        # Intrusion settings
        self.intrusion_cooldown = 5  # Seconds between intrusion alerts
        self.intrusion_threshold = 0.3  # Percentage of face area that must be different to trigger intrusion
        
    def start_camera(self):
        """Initialize and start the camera"""
        self.camera = cv2.VideoCapture(0)
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)
        
        # Check if camera opened successfully
        if not self.camera.isOpened():
            print("Error: Could not open camera.")
            return False
        
        return True
    
    def capture_reference_face(self):
        """Capture the candidate's face as a reference"""
        if not self.camera or not self.camera.isOpened():
            if not self.start_camera():
                return False
        
        print("\nCapturing reference face image...")
        print("Please look directly at the camera.")
        
        # Countdown
        for i in range(3, 0, -1):
            print(f"Taking picture in {i}...")
            time.sleep(1)
        
        # Capture frame
        ret, frame = self.camera.read()
        if not ret:
            print("Error: Could not capture frame.")
            return False
        
        # Convert to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces using OpenCV's Haar cascade
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        
        if len(faces) == 0:
            print("Error: No face detected. Please try again.")
            return False
        
        # Use the largest face if multiple are detected
        if len(faces) > 1:
            # Find the largest face by area
            largest_face_idx = 0
            largest_area = 0
            for i, (x, y, w, h) in enumerate(faces):
                area = w * h
                if area > largest_area:
                    largest_area = area
                    largest_face_idx = i
            
            x, y, w, h = faces[largest_face_idx]
        else:
            x, y, w, h = faces[0]
        
        # Create bounding box with padding
        x1 = max(0, x - self.bbox_padding)
        y1 = max(0, y - self.bbox_padding)
        x2 = min(frame.shape[1], x + w + self.bbox_padding)
        y2 = min(frame.shape[0], y + h + self.bbox_padding)
        
        # Store the bounding box
        self.face_bbox = (x1, y1, x2, y2)
        
        # Store the reference face region
        self.reference_face = frame[y1:y2, x1:x2].copy()
        
        # Save the reference image with bounding box for debugging
        debug_frame = frame.copy()
        cv2.rectangle(debug_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.imwrite("reference_face.jpg", debug_frame)
        
        print("Reference face captured successfully!")
        return True
    
    def start_monitoring(self):
        """Start monitoring for intrusions"""
        if self.reference_face is None or self.face_bbox is None:
            print("Error: Reference face not captured. Please capture a reference face first.")
            return False
        
        if self.monitoring:
            print("Already monitoring.")
            return True
        
        self.monitoring = True
        self.intrusion_detected = False
        self.intrusion_count = 0
        
        # Start monitoring in a separate thread
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        
        print("Intrusion monitoring started.")
        return True
    
    def stop_monitoring(self):
        """Stop monitoring for intrusions"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1.0)
            self.monitor_thread = None
        
        if self.camera and self.camera.isOpened():
            self.camera.release()
            self.camera = None
        
        print("Intrusion monitoring stopped.")
        return True
    
    def _monitor_loop(self):
        """Main monitoring loop that runs in a separate thread"""
        if not self.camera or not self.camera.isOpened():
            if not self.start_camera():
                self.monitoring = False
                return
        
        while self.monitoring:
            # Capture frame
            ret, frame = self.camera.read()
            if not ret:
                print("Error: Could not capture frame during monitoring.")
                time.sleep(0.1)
                continue
            
            # Extract the region of interest (face bounding box)
            x1, y1, x2, y2 = self.face_bbox
            roi = frame[y1:y2, x1:x2]
            
            # Check for intrusions
            intrusion = self._check_intrusion(roi)
            
            # Draw bounding box on the frame (green for normal, red for intrusion)
            color = (0, 0, 255) if intrusion else (0, 255, 0)
            debug_frame = frame.copy()
            cv2.rectangle(debug_frame, (x1, y1), (x2, y2), color, 2)
            
            # Save the frame if intrusion is detected (with rate limiting)
            if intrusion and time.time() - self.last_intrusion_time > self.intrusion_cooldown:
                self.intrusion_count += 1
                self.last_intrusion_time = time.time()
                cv2.imwrite(f"intrusion_{self.intrusion_count}.jpg", debug_frame)
                print(f"\n⚠️ INTRUSION DETECTED! ⚠️ (#{self.intrusion_count})")
            
            # Small delay to reduce CPU usage
            time.sleep(0.05)
    
    def _check_intrusion(self, current_roi):
        """Check if there's an intrusion in the current ROI compared to the reference"""
        if current_roi.shape[:2] != self.reference_face.shape[:2]:
            # Resize if dimensions don't match
            current_roi = cv2.resize(current_roi, (self.reference_face.shape[1], self.reference_face.shape[0]))
        
        # Convert to grayscale
        ref_gray = cv2.cvtColor(self.reference_face, cv2.COLOR_BGR2GRAY)
        curr_gray = cv2.cvtColor(current_roi, cv2.COLOR_BGR2GRAY)
        
        # Calculate absolute difference
        diff = cv2.absdiff(ref_gray, curr_gray)
        
        # Threshold the difference
        _, thresholded = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
        
        # Calculate the percentage of different pixels
        diff_percentage = np.sum(thresholded > 0) / (thresholded.shape[0] * thresholded.shape[1])
        
        # Return True if the difference exceeds the threshold
        return diff_percentage > self.intrusion_threshold
    
    def get_intrusion_status(self):
        """Get the current intrusion status"""
        return {
            "monitoring": self.monitoring,
            "intrusion_detected": self.intrusion_detected,
            "intrusion_count": self.intrusion_count
        }

# Example usage
if __name__ == "__main__":
    detector = IntrusionDetector()
    
    try:
        if detector.capture_reference_face():
            detector.start_monitoring()
            
            # Keep the script running
            print("Press Ctrl+C to stop monitoring.")
            while detector.monitoring:
                time.sleep(1)
    
    except KeyboardInterrupt:
        print("\nStopping monitoring...")
    
    finally:
        detector.stop_monitoring()
        print("Monitoring stopped. Exiting.") 