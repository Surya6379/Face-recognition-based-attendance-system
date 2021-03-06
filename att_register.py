import numpy as np
import matplotlib.pyplot as plt
import cv2
from keras_vggface.utils import preprocess_input
from keras_vggface.vggface import VGGFace
from scipy.spatial.distance import cosine
import csv
import datetime

    # def face_det_ext(img):
    #     face_det = MTCNN() 
    #     faces = face_det.detect_faces(img)
    #     co = list(faces[0]['box'])
    #     x1 , y1 , x2 , y2 = co[0] , co[1] , co[0]+co[2] , co[1]+co[3]
    #     cp = img[y1:y2,x1:x2]
    #     cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),3)
    #     return [cp,img]
def pre_proc(img1):
    img1 = cv2.resize(img1 , (224,224))
    img1 = np.expand_dims(img1,axis=0)
    return img1
def get_model_scores(faces):
    samples = np.array(faces, 'float32')
    samples = preprocess_input(samples, version=2)
    model = VGGFace(model='resnet50',
      include_top=False,
      input_shape=(224, 224, 3),
      pooling='avg')
    return model.predict(samples)
def write_csv(data):
    #"C:\\Users\\Sivaraman\\Desktop\\Py_projects\\face_rec_attendance\\real\\reg.csv"
    path = input("\nEnter path to save ids")
    with open(path ,"w", newline="") as csvf:
        dat = csv.writer(csvf)
        dat.writerow(data)
# def read_csv():
#     path = "C:\\Users\\Sivaraman\\Desktop\\Py_projects\\face_rec_attendance\\real\\reg.csv"
#     data = []
#     with open(path,"r",newline="") as csvf:
#         dat = csv.reader(csvf)
#         for row in dat:
#             data.append(row)
#     return data
# def write_csv_reg(data):
#     path = "C:\\Users\\Sivaraman\\Desktop\\Py_projects\\face_rec_attendance\\real\\att.csv"
#     with open(path ,"w", newline="") as csvf:
#         dat = csv.writer(csvf)
#         dat.writerow(data)
def register():
    face_cas = cv2.CascadeClassifier("C:\\Users\\Sivaraman\\Desktop\\Py_projects\\face_rec_attendance\\face_har.xml")
    ch="y"
    print("\nCLICK ON THE CAMERA WINDOW AND PRESS R TO REGISTER\n")
    while ch=="y":
        cap=cv2.VideoCapture(0)
        while True:
            ret,fimg=cap.read()
            #fimg = cv2.cvtColor(fimg,cv2.COLOR_BGR2RGB)
            gray = cv2.cvtColor(fimg, cv2.COLOR_BGR2GRAY)
            faces = face_cas.detectMultiScale(gray, 1.3, 5)
            for (x,y,w,h) in faces:
                cv2.rectangle(fimg,(x,y),(x+w,y+h),(255,0,0),2)
                img = fimg[y:y+h, x:x+w]
            cv2.imshow("win",fimg)
            k = cv2.waitKey(1)
            if k==ord('r'):
                break
        cap.release()
        cv2.destroyAllWindows()
        # img , fimg = face_det_ext(img)
        print("\nPreprocessing img\n")
        img = pre_proc(img)
        print("Getting model_scores\n")
        score = get_model_scores(img)
        print(np.array(score).shape,"\n")
        name = input("Give name ")
        scores.append([np.array(score),name])
        print("\nWriting on csv...")
        temp = np.append(score,[name])
        write_csv(temp)
        print("\nWrote on csv!...")
        ch = input("\nwanna add more faces? (y/n) ")
    print(scores)
    cv2.putText(fimg,"Registered", (10,50), cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0),3)
    cv2.imshow("win",fimg)
    k = cv2.waitKey(0)
    cv2.destroyAllWindows()  
# def scan():
#     face_cas = cv2.CascadeClassifier("C:\\Users\\Sivaraman\\Desktop\\Py_projects\\face_rec_attendance\\face_har.xml")
#     cap=cv2.VideoCapture(0)
#     while True:
#         ret,fimg=cap.read()
#         #fimg = cv2.cvtColor(fimg,cv2.COLOR_BGR2RGB)
#         gray = cv2.cvtColor(fimg, cv2.COLOR_BGR2GRAY)
#         faces = face_cas.detectMultiScale(gray, 1.3, 5)
#         for (x,y,w,h) in faces:
#             cv2.rectangle(fimg,(x,y),(x+w,y+h),(255,0,0),2)
#             img = fimg[y:y+h, x:x+w]
#         cv2.imshow("win",fimg)
#         k = cv2.waitKey(1)
#         if k==ord('r'):
#             break
#     cap.release()
#     cv2.destroyAllWindows()
#     # img , fimg = face_det_ext(img)
#     print("Preprocessing image... \n")
#     img = pre_proc(img)
#     print("Getting model scores...\n")
#     sc = get_model_scores(img)
#     flag = 0
#     print("Reading csv...\n")
#     data = read_csv()
#     for score in data:
#         cscore = np.array(score[:-1],'float32')
#         name = score[-1]
#         if(cosine(cscore,sc)<0.4):
#             print("hi ",name)
#             cv2.putText(fimg,"Hi "+name,(10,50), cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0),3)
#             flag=1
#             cv2.imshow("win",fimg)
#             k = cv2.waitKey(0)
#             cv2.destroyAllWindows()
#             now  = datetime.datetime.now()
#             data1 = [now.strftime("%Y-%m-%d"),now.strftime("%H:%M:%S"),name]
#             print("\nAdding attendance..\n")
#             write_csv_reg(data1)
#             break
#         if flag==0:
#             print("Wait a minute who are you")
#             cv2.putText(img1,"Face not registered", (10,50), cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0),3)
#             flag=1
#             cv2.imshow(img1)
#             k = cv2.waitKey(0)
#             cv2.destroyAllWindows()
scores = []
register()
