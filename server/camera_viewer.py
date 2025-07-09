# simple_viewer.py

from flask import Flask, Response, render_template_string
import zmq
import base64
import pickle
import numpy as np
import time
import threading
import queue
import cv2  # Add for debugging

import traceback


app = Flask(__name__)

# Image queue for each camera
frame_queue = queue.Queue(maxsize=2)  # Only keep latest frame

class VideoStreamer:
    def __init__(self, host, port):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.SUB)
        self.socket.setsockopt(zmq.CONFLATE, 1)  # Only get latest message
        self.socket.connect(f'tcp://{host}:{port}')
        self.socket.setsockopt(zmq.SUBSCRIBE, b"rgb_image")
        print(f"Connected to camera stream at {host}:{port}")
        
    def get_frame(self):
        try:
            print("Waiting for frame...")
            raw_data = self.socket.recv()
            print(f"Received data length: {len(raw_data)}")
            topic, data = raw_data.split(b' ', 1)
            print(f"Topic: {topic}")
            
            # Unpickle the data dictionary
            data_dict = pickle.loads(data)
            
            # The data structure in ZMQCameraPublisher has rgb_image as base64
            encoded_data = base64.b64decode(data_dict['rgb_image'])
            print(f"Decoded image size: {len(encoded_data)} bytes")
            
            # For debugging: save one frame to examine
            # with open("debug_frame.jpg", "wb") as f:
            #     f.write(encoded_data)
            
            return encoded_data
        except Exception as e:
            print(f"Error getting frame: {e}")
            traceback.print_exc()
            return None

def camera_worker(host, port):
    """Thread function that gets frames and puts them in the queue"""
    streamer = VideoStreamer(host, port)
    while True:
        try:
            frame = streamer.get_frame()
            if frame is not None:
                print(f"Got a valid frame, size: {len(frame)} bytes")
                # Clear queue and add new frame (always show latest)
                while not frame_queue.empty():
                    try:
                        frame_queue.get_nowait()
                    except:
                        pass
                frame_queue.put(frame)
            else:
                print("Frame was None")
            time.sleep(0.01)  # Small delay to prevent CPU hogging
        except Exception as e:
            print(f"Worker error: {e}")
            time.sleep(1)  # Delay on error

@app.route('/')
def index():
    """Render the home page with video stream"""
    return render_template_string("""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Robot Camera Viewer</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 0; padding: 20px; text-align: center; }
            h1 { color: #333; }
            .stream-container { margin: 20px auto; max-width: 800px; }
            img { width: 100%; border: 1px solid #ddd; border-radius: 4px; }
            #status { color: #777; margin-top: 10px; }
        </style>
        <script>
            // Add JavaScript to show connection status and reload on errors
            window.onload = function() {
                const img = document.querySelector('img');
                const status = document.getElementById('status');
                
                img.onload = function() {
                    status.textContent = 'Connected to camera stream';
                    status.style.color = 'green';
                };
                
                img.onerror = function() {
                    status.textContent = 'Error loading stream - check console';
                    status.style.color = 'red';
                    // Reload the image source after a delay
                    setTimeout(() => {
                        img.src = '/video_feed?' + new Date().getTime();
                    }, 2000);
                };
            };
        </script>
    </head>
    <body>
        <h1>Robot Camera Stream</h1>
        <div class="stream-container">
            <img src="/video_feed" alt="Camera Stream">
            <div id="status">Connecting...</div>
        </div>
    </body>
    </html>
    """)

@app.route('/video_feed')
def video_feed():
    """Video streaming route"""
    def generate():
        empty_frame_count = 0
        while True:
            try:
                # Get frame from queue
                frame = frame_queue.get(timeout=3.0)
                empty_frame_count = 0
                print(f"Sending frame to browser, size: {len(frame)} bytes")
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            except queue.Empty:
                empty_frame_count += 1
                print(f"Queue empty, count: {empty_frame_count}")
                # Create a blank image with text
                img = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(img, "Waiting for camera...", (50, 240), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                _, buffer = cv2.imencode('.jpg', img)
                empty_frame = buffer.tobytes()
                
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + empty_frame + b'\r\n')
                
                # If too many empty frames, try reconnecting
                if empty_frame_count > 10:
                    print("No frames received for too long, check camera connection")
            except Exception as e:
                print(f"Stream error: {e}")
                time.sleep(0.5)
    
    return Response(generate(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    # IMPORTANT: Update to match your actual camera stream port
    CAMERA_HOST = "10.29.155.27"  # Match the exact IP from your logs
    CAMERA_PORT = 10005  # This port is correct
    
    print(f"\nIMPORTANT: This script is connecting to: tcp://{CAMERA_HOST}:{CAMERA_PORT}")
    print("If this is not your camera's address, please update the CAMERA_HOST and CAMERA_PORT values.")
    print("The camera should be streaming using ZMQCameraPublisher from your realsense.py\n")
    
    # Start the camera thread
    camera_thread = threading.Thread(
        target=camera_worker, 
        args=(CAMERA_HOST, CAMERA_PORT),
        daemon=True
    )
    camera_thread.start()
    
    # Start the web server
    print("Starting web viewer at http://localhost:5000")
    print(f"Connecting to camera at {CAMERA_HOST}:{CAMERA_PORT}")
    print("Press Ctrl+C to exit")
    app.run(host='0.0.0.0', debug=False, threaded=True)