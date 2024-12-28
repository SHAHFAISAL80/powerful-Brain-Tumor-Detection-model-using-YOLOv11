from ultralytics import YOLO
import cv2

# Load the trained model
model = YOLO(r'Brain_tumor_model.pt')

# Path to the video file
video_path = r'vi.mp4'

# Start video capture from the video file
cap = cv2.VideoCapture(video_path)

# Get video properties
fps = cap.get(cv2.CAP_PROP_FPS)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define video writer
output_mp4 = cv2.VideoWriter('output_face_video.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("End of video or unable to read the frame.")
        break

    # Perform detection
    results = model(frame)

    # Annotate the frame manually
    y_offset_tracker = {}  # To track label offsets for each bounding box
    for detection in results[0].boxes.data:
        x1, y1, x2, y2, confidence, cls = map(int, detection[:6])
        label = 'Positive' if cls == 0 else 'Negative'  # Example: class 0 is Positive
        color = (255, 0, 255) if cls == 0 else (0, 255, 255)  # Pink for Positive, Yellow for Negative

        # Draw bounding box with thickness 4
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 4)

        # Avoid label overlap by tracking y-offsets for labels near each other
        y_offset = y_offset_tracker.get((x1, y1), y1 - 10)  # Default offset is above the box
        y_offset_tracker[(x1, y1)] = y_offset - 30  # Move up by 30 pixels for each new label

        # Draw label with thickness 4
        cv2.putText(frame, f'{label} {confidence:.2f}', (x1, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 4)

    # Display the annotated frame
    cv2.imshow('Brain_tumor_model', frame)

    # Write the annotated frame to output video
    output_mp4.write(frame)

    # Exit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
output_mp4.release()
cv2.destroyAllWindows()
