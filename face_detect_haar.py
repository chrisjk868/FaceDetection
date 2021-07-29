import cv2


def draw_rect(img, face_coordinates):
    for rect in face_coordinates:
        cv2.rectangle(img, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (0, 255, 0), 2)


def main():
    # haar cascade algorithm: pretrained model on face-frontal detection
    trained_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    webcam = cv2.VideoCapture(0)

    # iterate over frames
    while True:
        (success, frame) = webcam.read()
        # convert frames to greyscale for algorithm
        to_greyscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_coordinates = trained_data.detectMultiScale(to_greyscale, 1.05, 4, 0, [20, 20])
        draw_rect(frame, face_coordinates)
        cv2.imshow('Face Detection', frame)
        key = cv2.waitKey(10)
        if key == 81 or key == 113:
            break


# Testing git
if __name__ == '__main__':
    main()