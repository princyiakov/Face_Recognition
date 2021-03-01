import cv2
import face_recognition
import os


class FaceRecognition:
    def __init__(self, loc):
        self.loc = loc

    def load_known_faces(self):
        images = os.listdir(self.loc)
        img_names = []
        encoded_images = []
        for i in images:
            name = os.path.splitext(i)[0]
            img_names.append(name)
            img = face_recognition.load_image_file(os.path.join(self.loc, i))
            face_loc, encode_img = self.load_encode_loc(img, 1)
            encoded_images.append(encode_img)

        return img_names, encoded_images

    @staticmethod
    def load_encode_loc(image, flag):
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if flag == 1:  # To load known images having one person
            face_loc = 0
            encode_img = face_recognition.face_encodings(img)[0]
        else:  # To load unknown images which might have multi-person
            face_loc = face_recognition.face_locations(img)
            encode_img = face_recognition.face_encodings(img, face_loc)

        return face_loc, encode_img

    @staticmethod
    def get_match_facedis(knw_encode_img, curr_encode_img):
        matches = face_recognition.compare_faces(knw_encode_img, curr_encode_img)
        face_distance = face_recognition.face_distance(knw_encode_img, curr_encode_img)

        return matches, face_distance
