
# python -m PyQt5.uic.pyuic -x login.ui -o run.py
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QFileDialog
from run import Ui_MainWindow   
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import QThread, pyqtSignal, Qt
from PyQt5.QtCore import QTimer
from PyQt5.QtGui import QImage
import cv2

from keras.models import load_model
from time import sleep
from tensorflow.keras.utils import img_to_array
from keras.preprocessing import image
import cv2
import numpy as np


class Main_handle(Ui_MainWindow):
    def __init__(self):
        self.setupUi(MainWindow)
        self.btn_browser.clicked.connect(self.linkto)
        self.face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        # self.btn_start.clicked.connect(self.start_capture_video)
        # self.btn_start.clicked.connect(self.stop_capture_video)
        self.btn_start.clicked.connect(self.controlTimer)
        self.timer = QTimer()
        self.age_model = load_model('weights/best_model_multiclass_128.h5') 
        self.age_model1=load_model('weights/last_experiment5_1_300x300.h5') 
        self.gender_model = load_model('weights/gender_200x200.h5')
        self.timer.timeout.connect(self.detectface)
        self.scaling_factor=0.8
    
        
    
    #link to video
    def linkto(self):
        global link
        link = QFileDialog.getOpenFileName(filter='*.mp4 *.wav')
        self.line_Edit.setText(link[0])
        link = link[0]
    def config_frame(self,frame):
        
        # resize frame image
        frame = cv2.resize(frame, None, fx=self.scaling_factor, fy=self.scaling_factor, interpolation=cv2.INTER_AREA)
        # convert frame to GRAY format
       
        return frame



    #Detect face
    def detectface(self):
        
        gender_labels = ['FeMale', 'Male']
        # age_labels = ['0-3', '4-10','7-10','above 10']
        age_labels = ['0-3', '4-10','7-10','above 10']
        # age_labels = ['1-5', '6-10', '11-20', '21-30', '31-40','41-70',"70-100"]
        # read frame from video capture
        ret, frame = self.cap.read()
     
        frame=self.config_frame(frame)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        #Get image ready for prediction

        # detect rect faces
        face_rects = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        

        # for all detected faces
        for (x, y, w, h) in face_rects:
            # draw green rect on face
            roi_color=frame[y:y+h,x:x+w]
                
            roi_color_model1=cv2.resize(roi_color,(200,200),interpolation=cv2.INTER_AREA)
            roi_color2_model2=cv2.resize(roi_color,(300,300),interpolation=cv2.INTER_AREA)

            img_gray = cv2.cvtColor(roi_color_model1,cv2.COLOR_BGR2GRAY)

            img_gray=img_gray.astype('float')/255.0  #Scale
            img_gray = img_to_array(img_gray)
            img_gray=np.expand_dims(img_gray,axis=0) 
          


            roi_color=roi_color2_model2.astype('float')/255.0  #Scale
            roi_color = img_to_array(roi_color)
            roi_color=np.expand_dims(roi_color,axis=0)   
            
            #Age
            age_predict=self.age_model.predict(img_gray)[0]
            age_predict1=self.age_model1.predict(roi_color)
            array3=np.zeros((1,4))
            for i in range(3):
                array3[:,i]=age_predict[i]
            age_predict=age_predict1+array3 
            print(age_predict)
            age_label=age_labels[age_predict.argmax()] #Find the label 
           
            
            #Gender
            gender_predict = self.gender_model.predict(img_gray)[0] 
            gender_label=gender_labels[gender_predict.argmax()]

            gender_label_position=(x+150,y-20) #50 pixels below to move the label outside the face
            cv2.putText(frame,gender_label,gender_label_position,cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),3)
            age_label_position=(x,y-20)
            cv2.putText(frame,age_label,age_label_position,cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),3)
            #Draw bounding box
            cv2.rectangle(frame,(x,y),(x+w,y+h),(50,50,255),2) 
            

        # convert frame to RGB format
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # get frame infos
        height, width, channel = frame.shape
        step = channel * width

        # create QImage from RGB frame
        qImg = QImage(frame.data, width, height, step, QImage.Format_RGB888)
        # show frame in img_label

        self.Screen.setPixmap(QPixmap.fromImage(qImg))

    # start/stop timer
    def controlTimer(self):
        #select use cam or video
        if self.rad_cam.isChecked():
            sel = 0
        if self.rad_video.isChecked():
            sel = link
        # if timer is stopped
        if not self.timer.isActive():
            # create video capture
            self.cap = cv2.VideoCapture(sel)
            # start timer   
            self.timer.start(20)
            # update control_bt text
            self.btn_start.setText("STOP")
        else:
            # stop timer
            self.timer.stop()
            # release video capture
            self.cap.release()
            # update control_bt text
            self.btn_start.setText("START")
    
if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Main_handle()
    MainWindow.setWindowIcon(QtGui.QIcon('icon.png'))
    MainWindow.show()
    sys.exit(app.exec_())