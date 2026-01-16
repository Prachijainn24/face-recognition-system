import cv2

def detect_face():
    face_cascade = cv2.CascadeClassifier(
        "models/haarcascade_frontalface_default.xml"
    )

    img = cv2.imread("dataset/sample_faces/sample.jpg")
    if img is None:
        print("No image found. Add an image to dataset/sample_faces/")
        return

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

    cv2.imshow("Face Detection", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    detect_face()

