import cv2 as cv
import time
import os
print(os.getcwd())
from flask import Flask,request,render_template
app=Flask(__name__,template_folder="templates")
@app.route('/',methods=['GET'])
def index():
    return render_template('index.html')
@app.route('/index',methods=['GET'])
def about():
    return render_template('home.html')
@app.route('/image1',methods=['GET','POST'])
def image1():
    return render_template("index6.html")
faceproto = "opencv_face_detector.pbtxt"
faceModel = "opencv_face_detector_uint8.pb"

ageProto = "age_deploy.prototxt"
ageModel = "age_net.caffemodel"

genderProto = "gender_deploy.prototxt"
genderModel = "gender_net.caffemodel"
MODEL_MEAN_VALUES = (78.4263377603,87.7689143744,114.895847746)
ageList = ['(0-2)','(4-6)','(8-12)','(15-20)','(25-32)','(38-43)','(48-53)','(55-100)']
genderList = ['male','female']

ageNet = cv.dnn.readNetFromCaffe(ageProto,ageModel)
genderNet = cv.dnn.readNetFromCaffe(genderProto,genderModel)
faceNet = cv.dnn.readNet(faceModel,faceproto)

def getfaceBox(Net,frame,conf_threshold=0.7):
    frameopencvDnn = frame.copy()
    frameHight = frameopencvDnn.shape[0]
    frameWidth = frameopencvDnn.shape[1]
    blob =cv.dnn.blobfromimage(frameopencvDnn,1.0,(300,300),[104,117,123],True,False)
    
    ageNet.setInput(blob)
    detections = ageNet.forward()
    bboxes = []
    for i in range(detections.shape[2]):
        confidence = detections[0,0,i,2]
        if confidence > conf_threshold:
            x1 = int(detections[0,0,i,3]*  frameWidth)
            y1 = int(detections[0,0,i,4]*frameHight )
            x2 = int(detections[0,0,i,5]*  frameWidth)
            y2 = int(detections[0,0,i,6]* frameHight)
            bboxes.append([x1,y1,x2,y2])
            cv.rectangle(frameopencvDnn,(x1,y1),(x2,y2),(0,255,0),int(round(frameHight/150)),8)
    return frameopencvDnn, bboxes
@app.route('/predict',methods=['GET','POST'])
def image():
    if request.method == 'POST':
        print("inside image")
        f = request.files['image']
        
        basepath =os.path.dirname(__file__)
        file_path =os.path.join(basepath,'uploads','secure_filename'(f.filename))
        f.save(file_path)
        print(file_path)
    cap = cv.videocapture(file_path)
    padding =20
    while cv.waitkey(1)<0:
      t =time.time()
      hasFrame,frame = cap.read()
      if not hasFrame:
         cv.waitkey()
         break
      frameFace,bboxes = 'getFacebox'(faceNet,frame)
      if not bboxes:
        print("no face detected,checking next frame")
        continue
      for bbox in bboxes:
        face = frame[max(0,bbox[1]-padding):min(bbox[3]+padding,frame.shape[0]-1),
                     max(0,bbox[0]-padding):min(bbox[2]+padding,frame.shape[1]-1)]
      blob = cv.dnn.blobFromImage(face,1.0,(227,227),MODEL_MEAN_VALUES,swapRB=False)
      genderNet.setInput(blob)
      generpreds = genderNet.forwad()
      gender=genderList[ generpreds[0].argmax()]
      ageNet.setInput(blob)
      agePreds =ageNet.forward()
      age = ageList[agePreds[0].argmax()]
      label = "{},{}",format(gender,age) 
      cv.putText(frameFace,label,(bbox[0]-5, bbox[1]-10), cv.FONT_HERSHEY_SIMPLEX,0.75(0,0,255),2,cv.LINE_AA)
      cv.imshow("age gender demo",frameFace)
      if cv.waitkey(1) & 0xFF == ord('q'):
        break
      cap.release()
      cv.destroyallwindow()
    return render_template("index6.html")
@app.route('/upload',methods=['GET','POST'])
def predict():
    
    cap = cv.videocapture(0)
    padding =20
    while cv.waitkey(1)<0:
        t = time.time()
        hasFrame, frame =cap.read()
        if not hasFrame:
            cv.waitkey()
            break
        frameFace, bboxes ='getFacebox'(faceNet, frame)
        if not bboxes:
            print("no facedetected, checking next frame")
            continue
        face =frame[max(0,bboxes[1]-padding):min(bboxes[3]+padding,frame.shape[0]-1),
                    max(0,bboxes[0]-padding):min(bboxes[2]+padding, frame.shape[1]-1)]
        blob = cv.dnn.blobFromImage(face,1.0,(227,227),MODEL_MEAN_VALUES,swapRB=False)
        genderNet.setInput(blob)
        agePreds = ageNet.forward()
        age =ageList[agePreds[0].argmax()]
        label = "{},{}",format( genderNet,age)
        cv.putText(frameFace, label,(bboxes[0]-5,bboxes[1]-10),cv.FONT_HERSHEY_SIMPLEX,0.75,(0,0,255),2,cv.LINE_AA)
        cv.imshow("age gender demo",frameFace)
        if cv.waitkey(1)&0xff == ord('q'):
            break
        cap.release()
        cv.destroyAllwindows()
    return render_template("index.html")
    if __name__=='__main__':
        app.run(host='0.0.0.0',port=8000, debug=False)
