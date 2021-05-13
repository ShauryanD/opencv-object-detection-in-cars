import cv2

# Our files
#img_file = cv2.imread(D:\opencv\opencv-python-final-project\car_image.png)
#video = cv2.VideoCapture("D:\opencv\opencv-python-final-project\Tesla_Autopilot_Dashcam_Accident.mp4")
video = cv2.VideoCapture("D:\opencv\opencv-python-final-project\Pedestrians_1.mp4")

# Create car and pedestrian classifier
car_tracker = cv2.CascadeClassifier("D:\opencv\opencv-python-final-project\car_detector.xml")
pedestrian_tracker = cv2.CascadeClassifier("D:\opencv\opencv-python-final-project\haarcascades_fullbody.xml")

# Run the video till the frames end
while True:

    # Read the current frame
    (read_success, frame) = video.read()

    # If corrupt frame
    if read_success:
        # Convert the frame into grayscale
        grayscaled_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        break
    
    # Detect cars and pedestrians
    cars = car_tracker.detectMultiScale(grayscaled_frame)
    pedestrian = pedestrian_tracker.detectMultiScale(grayscaled_frame)


    # Draw rectangles around the cars
    for (x, y, w, h) in cars:
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0, 0, 255), 2)
    
    # Draw rectangles around the pedestrians
    for (x, y, w, h) in pedestrian:
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0, 255, 255), 2)

    # Display the image with cars spotted
    cv2.imshow("Car and Pedestrians detected", frame)

    # Don't autoclose
    key = cv2.waitKey(1)

    #Stop if Q is pressed
    if key==81 or key==133:
        break

#Release the video object
video.release()

print("Code Completed")


