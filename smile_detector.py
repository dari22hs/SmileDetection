"""
Smile detection using OpenCV's trained models
"""
import cv2


# Face and smile classifiers
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
smile_detector = cv2.CascadeClassifier('haarcascade_smile.xml')

# Grab webcam feed / Captura de la cámara web
webcam = cv2.VideoCapture(0)
# webcam = cv2.VideoCapture('video_instead_of_webcam.mp4')

while True:
    # Read current frame from webcam / Leer el cuadro actual de la cámara web
    successful_frame_read, frame = webcam.read()
    
    # If there's an error, abort / Si hay algún error, abortar
    if not successful_frame_read:
        break
    
    # Change to grayscale / Cambiar a escala de grises
    frame_grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces first / Primero detectar rostros
    faces = face_detector.detectMultiScale(frame_grayscale)
    
    # Run smile detection within each of those faces
    for (x, y, w, h) in faces:
        # Draw a rectanle around face / Dibujar rectángulo alrededor del rostro
        cv2.rectangle(frame, (x, y), (x+w, y+h), (100, 200, 50), 4)
        
        # Create the face sub-image (OpenCV allows to subindex like this. It's built on NumPy. Slices the n-dimensional array) / Crea la sub-imagen del rostro (OpenCV permite subindexar así. Está construido sobre NumPy. Corta un arreglo de n-dimensiones)
        face = frame[y:y+h, x:x+w]
        
        # Grayscale the face / Rostro en escala de grises
        face_grayscale = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        
        # Detect smiles in the face
        smile = smile_detector.detectMultiScale(face_grayscale, scaleFactor=1.7, minNeighbors=20)
        
        # Find smile on the face and draw a rectangle around the smile
        for (x_, y_, w_, h_) in smile:
            cv2.rectangle(face, (x_, y_), (x_ + w_, y_ + h_), (50, 50, 200), 3)
                
        # Label this face as 'SMILING'
        if len(smile) > 0:
            cv2.putText(frame, 'SMILING', (x, y+h+40), fontScale=3,
                        fontFace=cv2.FONT_HERSHEY_PLAIN, color=(255, 255, 255))
            
        # Show the current frame
        cv2.imshow('SMILE DETECTION', frame)
        
    # Stop if Q or q is pressed
    key = cv2.waitKey(1)
    if key == 81 or key == 113:
        break

# Release / Liberar
webcam.release()
cv2.destroyAllWindows()

print('Code completed!')
