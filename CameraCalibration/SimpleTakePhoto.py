from re import S
import cv2
import glob

# Input coordinates
coordinates = [[1200,100], [825,460], [0,500],[248,500]]

# Initiating Camera
camera = cv2.VideoCapture(1)
if not camera.isOpened():
    print("Cannot open camera")
    exit()

def crosshair(frame):
    # frame dimensions
    fheight = frame.shape[0]
    fwidth = frame.shape[1]
    # drawing a crosshair
    cv2.line(frame, (fwidth//2-100, fheight//2), (fwidth//2+100, fheight//2), (0, 255, 0))
    cv2.line(frame, (fwidth//2, fheight//2-100), (fwidth//2, fheight//2+100), (0, 255, 0))
# draws a dot at given coordinates in the image
# parameters: frame, coordinates, radius, color, line thickness (-1 will fill in circle)
def point(frame, coordinates):
    for coord in coordinates:
        cv2.circle(frame, coord, 5, (255,0,0), 3)

print("Camera on...")
result = True

# Press ESC to cancel. Press space to take photo. Press space to keep photo or press any other key to retake.
while(result):
    ret,frame = camera.read()
    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    
    # adding a crosshair
    crosshair(frame)
    # adding a point
    point(frame, coordinates)
    
    # display image
    cv2.imshow('Camera Feed',frame)
    
    # take photo
    take_photo = cv2.waitKey(1)
    if take_photo == 32:
        keep_photo = cv2.waitKey(0)
        if keep_photo == 32:
            cv2.imwrite(f"pixeltocoordinate_imagetesting/Distance_Testing" + "1" + ".png", frame)
            result = False
    elif take_photo == 27:
        break
camera.release()
cv2.destroyAllWindows()
