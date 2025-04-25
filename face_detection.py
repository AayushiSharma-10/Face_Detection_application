import cv2
import datetime

# loading Cascade to memory
face_cap = cv2.CascadeClassifier("C:/Users/Aayushi Sharma/anaconda3/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml")
smile_cap = cv2.CascadeClassifier("C:/Users/Aayushi Sharma/anaconda3/Lib/site-packages/cv2/data/haarcascade_smile.xml")

# live video capturing
video_cap=cv2.VideoCapture(0)       # taking video input from primary camera

while True:

    # reading frames and returning  'return code' and 'video frames'
    ret,video_data = video_cap.read()
    original_data = video_data.copy()
    
    # changing the color of image to gray
    col = cv2.cvtColor(video_data,cv2.COLOR_BGR2GRAY)

    # detect the face in live video
    faces = face_cap.detectMultiScale(
        col,
        scaleFactor=1.1,                            # specifying how much the image size is reduced at each image scale
        minNeighbors=5,                             # number of object dectected
        minSize=(30,30),                            # size of each window
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    
    # finding number of faces in the image
    print("Found {0} faces ".format(len(faces)))

    # drawing frame around faces
    for (x_axis,y_axis,width,heigth) in faces:
        cv2.rectangle(video_data,(x_axis,y_axis),(x_axis+width,y_axis+heigth),(255,0,0),4)
        
        # detecting region of interest for smile 
        face_roi = video_data[y_axis:y_axis+heigth , x_axis:x_axis+width]
        col_roi = col[y_axis:y_axis+heigth , x_axis:x_axis+width]
        smile = smile_cap.detectMultiScale(col_roi,1.3,25)
        
        # drawing frame around smile
        for (s_x_axis,s_y_axis,s_width,s_heigth) in smile :
            cv2.rectangle(face_roi,(s_x_axis,s_y_axis),(s_x_axis+s_width,s_y_axis+s_heigth),(0,0,255),2)
            
            time_stamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")     # noting the point of time image is captured     
            img_name = f'image-{time_stamp}.png'                                   # dynamically naming the image captured
            cv2.imwrite(img_name,original_data)                                    # saving the image

    # display the image
    cv2.imshow("video_live",video_data)

    # condition to stop the live video capturnig
    if cv2.waitKey(10) == ord(" "):
        break

# releasing the capturing video
video_cap.release()

# closing the windows
cv2.destroyAllWindows()