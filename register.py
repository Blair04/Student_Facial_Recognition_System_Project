import cv2
import os
import time

student_name = input("Enter Student Name: ").replace(" ", "_")
path = f'dataset/{student_name}'
os.makedirs(path, exist_ok=True)

cap = cv2.VideoCapture(0)
count = 0
total_needed = 100 
capturing = False 

print(f"--- REGISTRATION: {student_name} ---")
print("1. Press 'SPACE' to START automatic capture.")
print("2. Move your head slowly in a circle when it starts.")

while count < total_needed:
    ret, frame = cap.read()
    if not ret: break
    
    display_frame = frame.copy()
    status = "READY: Press SPACE" if not capturing else "CAPTURING..."
    color = (0, 255, 0) if not capturing else (0, 0, 255)
    
    cv2.putText(display_frame, f"Captured: {count}/{total_needed}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    cv2.putText(display_frame, status, (10, 450), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.imshow("Registering Student", display_frame)
    key = cv2.waitKey(1) & 0xFF
    
    if key == ord(' '):
        capturing = True

    if capturing:
        cv2.imwrite(f"{path}/{count}.jpg", frame)
        count += 1
        cv2.waitKey(50) # Controls speed of capture

    if key == ord('q'): break

cap.release()
cv2.destroyAllWindows()
print(f"Success! {count} images saved to {path}")