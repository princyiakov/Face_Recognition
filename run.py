import argparse
import os
from Utils import FaceRecognition
import cv2


def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(description='Face Recognition Script')
    parser.add_argument('--images', type=str,
                        default=os.path.join(current_dir, 'Images'), required=False,
                        help='Path to Known faces images')
    args = parser.parse_args()
    fr = FaceRecognition.FaceRecognition(args.images)
    kwn_names, kwn_encoding, kwn_loc = fr.load_known_faces()

    cap = cv2.VideoCapture(0)

    while True:
        flag, img = cap.read()
        img_transform = cv2.resize(img, (0, 0), fx=0.25, fy=0.25) # Resizing both axes by 1/4
        curr_face_loc, curr_encode_img = fr.load_encode_loc(img_transform)



if __name__ == '__main__':
    main()
