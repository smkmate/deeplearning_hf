import cv2
from modelloaders.race import RaceModel
from modelloaders.age import AgeModel
from modelloaders.gender import GenderModel
from models.ResnetRace import FaceClassificationModel

def face_recognition(racemodel, gendermodel, agemodel):
    # Loading haar cascade algorithm for face detection
    alg = "resources/face.xml"
    haar_cascade = cv2.CascadeClassifier(alg)

    # Opeining camera
    cam = cv2.VideoCapture(0)

    while True:
        # Getting image from the camera
        _, img = cam.read()
        text = "Face not detected"
        grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Detecting face with haar cascade on the gray image
        face = haar_cascade.detectMultiScale(grayImg, 1.3, 4)
        for (x, y, w, h) in face:
            # If a face was recognized the predict functions will be called in the models
            text = racemodel.predict(img[y:y + h, x:x + w])
            text += ", "
            text += gendermodel.predict(img[y:y + h, x:x + w])
            text += ", "
            text += agemodel.predict(img[y:y + h, x:x + w])

            # Drawing a rectangle around the face
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Printing either "Face not detected" or the outputs of the model on the image
        print(text)

        # Showing the processed image
        image = cv2.putText(img, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow("Face Detection", image)

        # If the esc button is pressed the loop will exit, and the app will be closed
        key = cv2.waitKey(10)
        if key == 27:
            break

    # Releasing camera, closing window
    cam.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':

    # Creating instances of the model loader classes
    racemodel = RaceModel()
    gendermodel = GenderModel()
    agemodel = AgeModel()

    face_recognition(racemodel, gendermodel, agemodel)
