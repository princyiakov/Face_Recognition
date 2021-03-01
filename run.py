import argparse
import os
from Utils import FaceRecognition
import cv2
import numpy as np


def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(description='Face Recognition Script')
    parser.add_argument('--images', type=str,
                        default=os.path.join(current_dir, 'Images'), required=False,
                        help='Path to Known faces images')
    args = parser.parse_args()
    fr = FaceRecognition.FaceRecognition(args.images)
    kwn_names, kwn_encoding = fr.load_known_faces()


    cap = cv2.VideoCapture(0)

    while True:
        flag, img = cap.read()
        img_transform = cv2.resize(img, (0, 0), fx=0.25, fy=0.25)  # Resizing both axes by 1/4
        curr_face_loc, curr_encode_img = fr.load_encode_loc(img_transform, 2)

        for face_loc, encode_img in zip(curr_face_loc, curr_encode_img):
            matches, face_dis = fr.get_match_facedis(kwn_encoding, encode_img)
            idx = np.argmin(face_dis)

            if matches[idx]:
                name = kwn_names[idx].upper()
                print(name)
                top, right, bottom, left = face_loc
                top, right, bottom, left = top*4, right*4, bottom*4, left*4
                cv2.rectangle(img, (left, top ),(right, bottom), (0,255,2), 2)
                cv2.rectangle(img, (left, bottom-25 ),(right, bottom), (0,255,2), cv2.FILLED)
                cv2.putText(img, name, (left+2 ,bottom-2), cv2.FONT_HERSHEY_COMPLEX, 1,(255,255,
                                                                                        255),2)

        cv2.imshow('Webcam', img)
        cv2.waitKey(1)


if __name__ == '__main__':
    main()
