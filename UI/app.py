import sys
import cv2
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QPushButton, QFileDialog, QVBoxLayout, QStackedWidget, QLineEdit
from PyQt5.QtGui import QPixmap,QImage,QCursor
from PyQt5.QtCore import Qt, QPoint,QTimer
import numpy as np

import os
# comment out below line to enable tensorflow logging outputs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import time
import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
from absl import app, flags, logging
from absl.flags import FLAGS
import core.utils as utils
from core.yolov4 import filter_boxes
from tensorflow.python.saved_model import tag_constants
from core.config import cfg
from PIL import Image
import matplotlib.pyplot as plt
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

# deep sort imports
from deep_sort import preprocessing, nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet

import xlsxwriter

class VideoPlayer(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle('Vehicle Speed Estimation System')
        self.setGeometry(100, 100, 1920, 1080)

        self.setStyleSheet('background-color: #10034a;')
                
        self.text_lbl = QLabel(self)
        self.text_lbl.setGeometry(0, 0, 1920, 90)
        self.text_lbl.setStyleSheet('background-color: #000000; font-size: 40px; color: #FFFFFF')
        self.text_lbl.setAlignment(Qt.AlignCenter)
        self.text_lbl.setText('Vehicle Speed Estimation System')
        
        self.stacked_widget = QStackedWidget(self)
        self.stacked_widget.setGeometry(30, 100, 1920*0.75, 1080*0.75)
        #self.stacked_widget.setAlignment(Qt.AlignCenter)
        
        self.lbl = QLabel(self.stacked_widget)
        self.lbl.setAlignment(Qt.AlignCenter)
        self.lbl.setScaledContents(True)

        self.result_lbl = QLabel(self.stacked_widget)
        self.result_lbl.setAlignment(Qt.AlignCenter)
        self.result_lbl.setScaledContents(True)

        self.stacked_widget.addWidget(self.lbl)
        self.stacked_widget.addWidget(self.result_lbl)
        
        
        self.text_upload = QLabel(self)
        self.text_upload.setGeometry(700, 300, 600, 100)
        self.text_upload.setStyleSheet('font-size: 30px; color: #FFFFFF')
        self.text_upload.setAlignment(Qt.AlignCenter)
        self.text_upload.setWordWrap(True)
        self.text_upload.setText('Please Upload a video to get Started.\n\nSupported File Types : .mov,  .mp4,  .avi')
    
        self.btn = QPushButton('Choose Video', self)
        self.btn.setStyleSheet("background-color : white; font-size: 18px")
        self.btn.setGeometry(860, 450, 200, 40)
        self.btn.clicked.connect(self.choose_video)
        
        self.homographytext = QLabel(self)
        self.homographytext.setGeometry(1470, 200, 550, 150)
        self.homographytext.setStyleSheet('font-size: 25px; color: #FFFFFF; background-color: black')
        self.homographytext.setAlignment(Qt.AlignCenter)
        self.homographytext.setWordWrap(True)
        self.homographytext.setText('Please mark the points for Homography mapping\n\n')
        self.homographytext.hide()
        
        self.done_btn = QPushButton('Done', self)
        self.done_btn.setStyleSheet("background-color : white")
        self.done_btn.setGeometry(1645, 900, 100, 30)
        self.done_btn.clicked.connect(self.send_points)
        self.done_btn.hide()
        
        
        self.submit_btn = QPushButton('Submit', self)
        self.submit_btn.setStyleSheet("background-color : white")
        self.submit_btn.setGeometry(1645, 800, 100, 30)
        self.submit_btn.hide()
        
        self.repeat_btn = QPushButton('Next video', self)
        self.repeat_btn.setStyleSheet("background-color : white")
        self.repeat_btn.setGeometry(1595, 200, 200, 30)
        self.repeat_btn.clicked.connect(self.repeat_video)
        self.repeat_btn.hide()

        self.exit_btn = QPushButton('Exit', self)
        self.exit_btn.setStyleSheet("background-color : red; color : white")
        self.exit_btn.setGeometry(1645, 250, 100, 30)
        self.exit_btn.clicked.connect(self.exit_app)
        self.exit_btn.hide()

        self.input_width = QLineEdit(self)
        self.input_width.setStyleSheet("background-color : white; font-size: 20px")
        self.input_width.setGeometry(1520,620,200,40)
        self.input_width.setPlaceholderText("Enter width in centimeter")
        self.input_width.hide()
        
        self.input_height = QLineEdit(self)
        self.input_height.setStyleSheet("background-color : white; font-size: 20px")
        self.input_height.setGeometry(1520,680,200,30)
        self.input_height.setPlaceholderText("Enter height")
        self.input_height.hide()
        
        
        self.points = []
        self.file_path = ''
        
        
    def choose_video(self):
        self.file_path, _ = QFileDialog.getOpenFileName(self, "Open Video", "", "Video Files (*.mp4 *.avi *.mkv *.mov)")
        if self.file_path:
            self.text_upload.hide()
            cap = cv2.VideoCapture(self.file_path)
            ret, self.frame = cap.read()
            cap.release()
            self.btn.hide()
            if ret:
                self.frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
                self.frame = cv2.resize(self.frame,(self.frame.shape[0],self.frame.shape[1]),interpolation = cv2.INTER_AREA)
                qimg = QPixmap.fromImage(QImage(self.frame, self.frame.shape[1], self.frame.shape[0], QImage.Format_RGB888))
                qimg_resized = qimg.scaled(self.lbl.width(), self.lbl.height())
                self.lbl.setPixmap(qimg_resized)
                self.stacked_widget.setCurrentWidget(self.lbl)
                self.homographytext.show()
                self.lbl.mousePressEvent = self.get_points
            else:
                self.lbl.setText('Could not load video')

    def get_points(self, event):
        x = event.pos().x()
        y = event.pos().y()
        self.frame = cv2.resize(self.frame,(int(1920*0.75),int(1080*0.75)),interpolation=cv2.INTER_AREA)
        self.frame = cv2.circle(self.frame,(x,y),3,(204, 255, 102),-1)
        qimg = QPixmap.fromImage(QImage(self.frame, self.frame.shape[1], self.frame.shape[0], QImage.Format_RGB888))
        qimg_resized = qimg.scaled(self.lbl.width(), self.lbl.height())
        self.lbl.setPixmap(qimg_resized)
        self.stacked_widget.setCurrentWidget(self.lbl)
        self.points.append((x, y))
        
        print(f"Added point ({x}, {y})")
        if len(self.points) < 4:
            print("Got all points!")
            print(self.points)
        
        else:
            self.homographytext.setGeometry(1470, 200, 450, 400)
            self.homographytext.setText('You have marked the four points: \n\n{}\n\n{}\n\n{}\n\n{}'.format(self.points[0],self.points[1],self.points[2],self.points[3]))
            self.input_width.show()
            self.input_height.show()
            self.submit_btn.show()
            self.submit_btn.clicked.connect(self.get_input)
            
    def get_input(self):
        width = int(self.input_width.text())
        height = int(self.input_height.text())
        print(width)
        print(height)
        self.done_btn.show()
    
    
    
    def send_points(self):
        self.input_height.hide()
        self.input_width.hide()
        self.submit_btn.hide()
        self.homographytext.hide()
        self.done_btn.hide()
        print("Sending points and file path to main function...")
        print(f"Points: {self.points}")
        print(f"File path: {self.file_path}")
        
        weights = './checkpoints/custom-416'
        iou = 0.40
        score = 0.50

        w1 = 630
        h1 = 1200
        
        # Definition of the parameters
        max_cosine_distance = 0.4
        nn_budget = None
        nms_max_overlap = 1.0
        
        # initialize deep sort
        model_filename = 'model_data/mars-small128.pb'
        encoder = gdet.create_box_encoder(model_filename, batch_size=1)
        
        # calculate cosine distance metric
        metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
        
        # initialize tracker
        tracker = Tracker(metric)

        # load configuration for object detector
        config = ConfigProto()
        config.gpu_options.allow_growth = True
        session = InteractiveSession(config=config)
        STRIDES, ANCHORS, NUM_CLASS, XYSCALE = utils.load_config()
        input_size = 416
        video_path = self.file_path

        saved_model_loaded = tf.saved_model.load(weights, tags=[tag_constants.SERVING])
        infer = saved_model_loaded.signatures['serving_default']
            

        # Initialise a frame counter and get the current time for FPS calculation purposes.
        frame_id = 0
        time_start = time.time()

        # Track counter
        tracked = [0] * 100000

        # Time counter
        elapseTime = [0] * 100000

        #Distance counter
        dist = [[0]*10000] * 100000

        #Entry track
        entry = [0] * 100000

        #Frame travelled
        frameTravel = [0] * 100000

        #Distance in 10 frames
        totaldist = [0] * 1000

        #Center coordinate:
        center_xT = [[0]*10000] * 10000
        center_yT = [[0]*10000] * 10000

        count = 0
        speed = [0] * 10000

        
        FONT = cv2.FONT_HERSHEY_COMPLEX
        FONT_SCALE = 0.75
        FONT_THICKNESS = 2
        
        
        #Create an excel file
        outWorkBook = xlsxwriter.Workbook("results.xlsx")
        outSheet = outWorkBook.add_worksheet("VEHICLE")

        #Create Headers
        outSheet.write("A1","TRACK ID")
        outSheet.write("B1","Speed")

        #Counter for row update
        roww = 2
        
        # begin video capture
        try:
            vid = cv2.VideoCapture(int(video_path))
        except:
            vid = cv2.VideoCapture(video_path)

        out = None

        frame_num = 0
        
        width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # while video is running
        while True:
            return_value, frame = vid.read()
            if return_value:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame,(int(1920*0.75),int(1080*0.75)),interpolation=cv2.INTER_AREA)
            else:
                print('Video has ended or failed, try a different video format!')
                self.repeat_btn.show()
                self.exit_btn.show()
                break
            
            
            #Homography Transformation
            points1 = self.points
            points1 = np.float32(points1[0:4])
            points2 = np.float32([[0,0],[w1,0],[0,h1],[w1,h1]])

            #ROI Selection
            area1 = np.array([points1[0],points1[1],points1[3],points1[2]],np.int32)
            cv2.polylines(frame,[area1],True,(0,255,0),3)          

            #Transformation
            matrix,status = cv2.findHomography(points1,points2)
            
            frame_num +=1
            frame_id +=1
            
            #print('Frame #: ', frame_num)
            frame_size = frame.shape[:2]
            image_data = cv2.resize(frame, (input_size, input_size))
            image_data = image_data / 255.
            image_data = image_data[np.newaxis, ...].astype(np.float32)
            start_time = time.time()

            
            batch_data = tf.constant(image_data)
            pred_bbox = infer(batch_data)
            for key, value in pred_bbox.items():
                boxes = value[:, :, 0:4]
                pred_conf = value[:, :, 4:]

            boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
                boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
                scores=tf.reshape(
                    pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
                max_output_size_per_class=50,
                max_total_size=50,
                iou_threshold=iou,
                score_threshold=score
            )

            # convert data to numpy arrays and slice out unused elements
            num_objects = valid_detections.numpy()[0]
            bboxes = boxes.numpy()[0]
            bboxes = bboxes[0:int(num_objects)]
            scores = scores.numpy()[0]
            scores = scores[0:int(num_objects)]
            classes = classes.numpy()[0]
            classes = classes[0:int(num_objects)]

            # format bounding boxes from normalized ymin, xmin, ymax, xmax ---> xmin, ymin, width, height
            original_h, original_w, _ = frame.shape
            bboxes = utils.format_boxes(bboxes, original_h, original_w)

            # store all predictions in one parameter for simplicity when calling functions
            pred_bbox = [bboxes, scores, classes, num_objects]

            # read in all class names from config
            class_names = utils.read_class_names(cfg.YOLO.CLASSES)

            # by default allow all classes in .names file
            allowed_classes = list(class_names.values())

            # loop through objects and use class index to get class name, allow only classes in allowed_classes list
            names = []
            deleted_indx = []
            for i in range(num_objects):
                class_indx = int(classes[i])
                class_name = class_names[class_indx]
                if class_name not in allowed_classes:
                    deleted_indx.append(i)
                else:
                    names.append(class_name)
            names = np.array(names)
            count = len(names)
            
            # delete detections that are not in allowed_classes
            bboxes = np.delete(bboxes, deleted_indx, axis=0)
            scores = np.delete(scores, deleted_indx, axis=0)

            # encode yolo detections and feed to tracker
            features = encoder(frame, bboxes)
            detections = [Detection(bbox, score, class_name, feature) for bbox, score, class_name, feature in zip(bboxes, scores, names, features)]

            #initialize color map
            cmap = plt.get_cmap('tab20b')
            colors = [cmap(i)[:3] for i in np.linspace(0, 1, 20)]

            # run non-maxima supression
            boxs = np.array([d.tlwh for d in detections])
            scores = np.array([d.confidence for d in detections])
            classes = np.array([d.class_name for d in detections])
            indices = preprocessing.non_max_suppression(boxs, classes, nms_max_overlap, scores)
            detections = [detections[i] for i in indices]       

            # Call the tracker
            tracker.predict()
            tracker.update(detections)

            # update tracks
            for track in tracker.tracks:
                if not track.is_confirmed() or track.time_since_update > 1:
                    continue 
                bbox = track.to_tlbr()
                class_name = track.get_class()
            
                #Get the coordinates for bottom center of bbox:
                centerCor = (int((bbox[0] + bbox[2])/2),int(bbox[3]))
                centerList =  np.array([[int((bbox[0] + bbox[2])/2),int(bbox[3])]], dtype=np.float32)    

                result = cv2.pointPolygonTest(area1,tuple([int((bbox[0]+bbox[2])/2),int(bbox[3])]),False)    
                
                if result >= 0.0:
                    # draw bbox on screen
                    color = colors[int(track.track_id) % len(colors)]
                    color = [i * 255 for i in color]
                    cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
                    cv2.rectangle(frame, (int(bbox[0]), int(bbox[1]-30)), (int(bbox[0])+(len(class_name)+len(str(track.track_id)))*17, int(bbox[1])), color, -1)
                    cv2.putText(frame, class_name + "-" + str(track.track_id),(int(bbox[0]), int(bbox[1]-10)),0, 0.75, (255,255,255),2)
                    cv2.circle(frame,centerCor,2,(0,255,0),-1)
                    trackNo = tracked[track.track_id]
                
                    #Conversion from homography coordinates to cartesian coordinates
                    
                    centerTrans = cv2.perspectiveTransform(centerList.reshape(-1, 1, 2),matrix)
                    center_xT[track.track_id][trackNo] = centerTrans[0][0][0]
                    center_yT[track.track_id][trackNo] = centerTrans[0][0][1]
                    
                    cv2.rectangle(frame,(int(bbox[2]),int(bbox[1]-50)),(int(bbox[2]+150),int(bbox[1])),color,-1)
                    cv2.putText(frame, str(round(speed[track.track_id],2)), (int(bbox[2] + 55),int(bbox[1] - 20)), FONT, FONT_SCALE, (255, 255, 255), 2)
                    
                    #Update Track counter.
                    if (tracked[track.track_id] == 0):
                        tracked[track.track_id] = tracked[track.track_id] + 1
                        entry[track.track_id] = frame_id
                        
                    else:
                        frameTravel[track.track_id] = frame_id - entry[track.track_id]
                        
                        tracked[track.track_id] = tracked[track.track_id] + 1
                        elapseTime[track.track_id] = elapseTime[track.track_id]+1
                    
                        dist[track.track_id][trackNo] = int(((center_xT[track.track_id][trackNo] - center_xT[track.track_id][(trackNo-1)]) ** 2 + (center_yT[track.track_id][trackNo] - center_yT[track.track_id][(trackNo-1)]) ** 2) ** 0.5)
                    
                        if (trackNo % 5) != 0:
                            totaldist[track.track_id] += dist[track.track_id][trackNo]
                            
                        else:
                            speed[track.track_id] = totaldist[track.track_id] * (3600*30)/(100*1000)/(frameTravel[track.track_id])                    
                            totaldist[track.track_id]=0
                            
                            entry[track.track_id] = frame_id
                            cv2.rectangle(frame,(int(bbox[2]),int(bbox[1]-50)),(int(bbox[2]+150),int(bbox[1])),color,-1)
                            cv2.putText(frame, str(round(speed[track.track_id],2)), (int(bbox[2] + 55),int(bbox[1] - 20)), FONT, FONT_SCALE, (255, 255, 255), 2)
                            time.sleep(1)
                            print('{}  {}'.format(int(center_xT[track.track_id][trackNo]),int(center_yT[track.track_id][trackNo])))
                            
                            print('{}  {}  {}'.format(track.track_id,trackNo,speed[track.track_id]))  
                            
                            col1 = "A" + str(roww)
                            col2 = "B" + str(roww)
                            outSheet.write(col1,track.track_id)
                            outSheet.write(col2,round(speed[track.track_id],2))
                            roww += 1
    
            # calculate frames per second of running detections
            fps = 1.0 / (time.time() - start_time)
            #print("FPS: %.2f" % fps)
            resultimg = np.asarray(frame)
            #result = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            
            # display processed frame
            qimg = QPixmap.fromImage(QImage(frame, frame.shape[1], frame.shape[0], QImage.Format_RGB888))
            qimg_resized = qimg.scaled(self.stacked_widget.width(), self.stacked_widget.height())
            self.result_lbl.setPixmap(qimg_resized)
            self.stacked_widget.setCurrentWidget(self.result_lbl)

            
            # update UI
            QApplication.processEvents()
            
        vid.release()     
            
        outWorkBook.close()
        cv2.destroyAllWindows()

    def repeat_video(self):
        self.points = []
        self.file_path = ''
        self.lbl.clear()
        self.result_lbl.clear()
        self.btn.show()
        self.repeat_btn.hide()
        self.exit_btn.hide()

    def exit_app(self):
        sys.exit()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    player = VideoPlayer()
    player.show()
    sys.exit(app.exec_())
