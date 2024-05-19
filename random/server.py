from flask import Flask, Response
import cv2
import base64

app = Flask(__name__)

# Initialize the camera
cap = cv2.VideoCapture(0)

def generate_frames():
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert frame to base64
        _, buffer = cv2.imencode('.jpg', frame)
        frame_base64 = base64.b64encode(buffer).decode('utf-8')
        print(f"Frame: {frame_base64[:10]}...")

        yield f"data:image/jpeg;base64,{frame_base64}"

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True, port=8080)