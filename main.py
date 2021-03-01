import cv2
import face_recognition

# Load the image
img_princy = face_recognition.load_image_file('Images/Princy.jpg')
img_princy = cv2.cvtColor(img_princy, cv2.COLOR_BGR2RGB)
img_test = face_recognition.load_image_file('Images/download.png')
img_test = cv2.cvtColor(img_test, cv2.COLOR_BGR2RGB)

# Face Encodings
face_loc = face_recognition.face_locations(img_princy)[0]
encode_princy = face_recognition.face_encodings(img_princy)[0]
cv2.rectangle(img_princy, (face_loc[3], face_loc[0]), (face_loc[1], face_loc[2]), (255, 0, 255), 2)

face_loc_test = face_recognition.face_locations(img_test)[0]
encode_test = face_recognition.face_encodings(img_test)[0]
cv2.rectangle(img_test, (face_loc_test[3], face_loc_test[0]), (face_loc_test[1], face_loc_test[2]),
              (255, 0, 255), 2)

# View Comparisions
results = face_recognition.compare_faces([encode_princy], encode_test)
face_distance = face_recognition.face_distance([encode_princy], encode_test)
print(results, face_distance)
cv2.putText(img_test, f'{results} {round(face_distance[0], 2)}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
            1, (0, 0, 255), 2)

cv2.imshow('Princy', img_princy)
cv2.imshow('Test', img_test)

cv2.waitKey(
    0)  # waitKey(0) will display the window infinitely until any keypress (it is suitable for image display).
