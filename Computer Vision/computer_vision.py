import cv2
import mediapipe as mp

# Initialize mediapipe face mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=2)

# Drawing styles
mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

# Open webcam
cap = cv2.VideoCapture(1)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process frame
    results = face_mesh.process(rgb_frame)

    # Draw face mesh
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=drawing_spec,
                connection_drawing_spec=drawing_spec
            )

    # Show frame
    cv2.imshow('Facial Mapping (Face Mesh)', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


# import cv2
# # Loads Haar cascade for face detection
# face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# cap = cv2.VideoCapture(1)

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     faces = face_cascade.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=5)

#     for (x, y, w, h) in faces:
#         cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

#     cv2.imshow('Face Detection', frame)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()

# import cv2
# # Captures video using default mac camera
# cap = cv2.VideoCapture(1)

# # Loop to read camera
# while True:
#   ret, frame = cap.read()
#   if not ret:
#     break
  
#   gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#   cv2.imshow('Live Camera', gray) # replace gray with frame for normal Camera

#   if cv2.waitKey(1) & 0xFF == ord('q'):
#     break

# # Release the camera and close all OpenCV windows
# cap.release()
# cv2.destroyAllWindows()