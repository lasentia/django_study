from django.shortcuts import render
from django.http import StreamingHttpResponse
import cv2
import torch
from ultralytics import YOLO ## 수정하기

# Create your views here.
def index(request):
    return render(request, 'index.html')

def yolomodel(video_path, model): ## 수정하기
    cap = cv2.VideoCapture(video_path)

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break
    
        # Preprocess the image
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_tensor = torch.from_numpy(img.transpose((2,0,1))).float().div(255.0).unsqueeze(0)

        # Make a prediction
        with torch.no_grad():
            output = model(img_tensor)[0]

        # Draw the bounding boxes on the image
        for i, det in enumerate(output):
            x1y1, x2y2, conf, cls = det.tolist()
            if conf > 0.5:
                x1, y1 = int(x1y1[0]), int(x1y1[1])
                x2, y2 = int(x2y2[0]), int(x2y2[1])
                label = f'class {int(cls)}'
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255,0,0), 2)
                cv2.putText(frame, label, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 2)

        # Convert the image to a byte string
        retval, buffer = cv2.imencode('.jpg', frame)
        image_bytes = buffer.tobytes()

        # Yield the byte string as a video frame
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + image_bytes + b'\r\n')

    cap.release()

def video(request):
    # Set the video path
   
    video_path = 'D:/django/yolov8_tracking/walk.mp4'

    # Load the YOLO model
    weights_file = 'D:/django/yolov8_tracking/yolov8n.pt'
    # model = torch.load('D:/django/yolov8_tracking/track.py', map_location=device)
    # model = torch.hub.load('yolov8_tracking/track', 'yolov8n') 
    model = YOLO('yolov8n.pt')
    # D:\django\yolov8_tracking\track.py
    
    # Set the device for model prediction
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model.to(device)

    # Call the yolomodel function and return the video feed as a HTTP response
    return StreamingHttpResponse(yolomodel(video_path, model), content_type='multipart/x-mixed-replace; boundary=frame')
