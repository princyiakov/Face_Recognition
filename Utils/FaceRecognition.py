import cv2
import face_recognition
import os


class FaceRecognition:
    def __init__(self, loc):
        self.loc = loc

    def load_known_faces(self):
        images = os.listdir(self.loc)
        img_names = []
        loc_faces = []
        encoded_images = []
        for i in images:
            name = os.path.splitext(i)[0]
            img_names.append(name)
            img = face_recognition.load_image_file(os.path.join(self.loc, i))
            face_loc, encode_img = self.load_encode_loc(img)
            encoded_images.append(encode_img)
            loc_faces.append(face_loc)

        return img_names, encoded_images, loc_faces

    def load_encode_loc(self, image):
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        face_loc = face_recognition.face_locations(img)
        encode_img = face_recognition.face_encodings(img, face_loc)

        return face_loc, encode_img



