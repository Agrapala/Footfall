import cv2
from age_gender_logger import log_age_gender
import face_recognition
import numpy as np


def faceBox(faceNet, frame):
    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]
    

    blog=cv2.dnn.blobFromImage(frame, 1.0, (227,227), [104, 117, 123], swapRB=False)
    faceNet.setInput(blog)
    detections = faceNet.forward()
    bboxs = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.7:
            detections[0,0,i,3] * frameWidth
            detections[0,0,i,4] * frameHeight
            x1 = int(detections[0, 0, i, 3]*frameWidth)
            y1 = int(detections[0, 0, i, 4]*frameHeight)
            x2 = int(detections[0, 0, i, 5]*frameWidth)
            y2 = int(detections[0, 0, i, 6]*frameHeight)
            bboxs.append([x1, y1, x2, y2])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 1)
    return frame, bboxs

faceProto = "opencv_face_detector.pbtxt"
faceModel = "opencv_face_detector_uint8.pb"

ageProto = "age_deploy.prototxt"
ageModel = "age_net.caffemodel"

genderProto = "gender_deploy.prototxt"
genderModel = "gender_net.caffemodel"

faceNet = cv2.dnn.readNet(faceModel, faceProto)
ageNet = cv2.dnn.readNet(ageModel, ageProto)
genderNet = cv2.dnn.readNet(genderModel,genderProto)

ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60+)']
genderList = ['Male','Female']
mean_value = (78.4263377603, 87.7689143744, 114.895847746)

video= cv2.VideoCapture(0)
padding = 20
logged_encodings = []
SIMILARITY_THRESHOLD = 0.6
while True:
    ret, frame = video.read()
    frame,bboxs = faceBox(faceNet, frame)
    for i, bbox in enumerate(bboxs) :
        face = frame[max(0, bbox[1]-padding):min(bbox[3]+padding, frame.shape[0]-1),max(0, bbox[0]-padding):min(bbox[2]+padding, frame.shape[1]-1)]
        blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), mean_value, swapRB=False)  
        genderNet.setInput(blob)
        genderProd = genderNet.forward()
        gender= genderList[genderProd[0].argmax()]
        ageNet.setInput(blob)
        ageProd = ageNet.forward()
        age = ageList[ageProd[0].argmax()]
        label = "{},{}".format(gender, age)
        cv2.rectangle(frame, (bbox[0], bbox[1]-30), (bbox[2], bbox[1]), (0, 255, 0), -1)
        cv2.putText(frame, label, (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        # Compute face encoding for uniqueness
        rgb_face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        encodings = face_recognition.face_encodings(rgb_face)
        if encodings:
            encoding = encodings[0]
            is_new = True
            for logged in logged_encodings:
                dist = np.linalg.norm(encoding - logged)
                if dist < SIMILARITY_THRESHOLD:
                    is_new = False
                    break
            if is_new:
                bbox_str = f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}"
                conf = max(float(genderProd[0].max()), float(ageProd[0].max()))
                log_age_gender(len(logged_encodings), age, gender, conf, bbox_str, "Camera 0")
                logged_encodings.append(encoding)
    cv2.imshow("Age-gender",frame)
    k=cv2.waitKey(1)
    if k==ord('q'):
        break
    
video.release()
cv2.destroyAllWindows()