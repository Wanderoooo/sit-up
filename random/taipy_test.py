from taipy import Gui
import cv2
import threading     
import base64

page = """
## Video Stream
<|/placeholder.jpg|image|id=video-stream|>
"""

def update_image(frame_base64):
    taipy.update('video-stream', {'content': f'data:image/jpeg;base64,{frame_base64}'})

def generate_frames():
    while True:
        print("please?")
        ret, frame = cap.read()
        if not ret:
            continue
        # Convert frame to base64
        _, buffer = cv2.imencode('.jpg', frame)
        frame_base64 = base64.b64encode(buffer).decode('utf-8')
        print(f"Frame: {frame_base64[:10]}...")
        update_image(frame_base64)

        yield f"data:image/jpeg;base64,{frame_base64}"

if __name__ == "__main__":
    cap = cv2.VideoCapture(0)

    generate_frames();


    # Initialize Taipy GUI
    taipy = Gui(page)

    # Start a separate thread for frame generation
    frame_thread = threading.Thread(target=generate_frames)
    frame_thread.start()

    print("Running Taipy GUI...")

    # Run the Taipy GUI (this will block until the GUI is closed)
    taipy.run(debug=True)

    # Wait for the frame generation thread to finish (optional)
    frame_thread.join()