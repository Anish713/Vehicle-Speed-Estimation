import cv2
import argparse
import os

parser = argparse.ArgumentParser(description='Frame Sampling for 1 frame per second')
parser.add_argument('-v', type=str)
parser.add_argument('-d', type=str)
args = parser.parse_args()

video = cv2.VideoCapture(args.v)

os.chdir(args.d)
currentframe=1
fps = int(video.get(cv2.CAP_PROP_FPS))
print('FPS Of Video = ' + str(fps))

while(video.isOpened()):
    ret,frame = video.read() 
    if ret:
        if(currentframe%fps==0):
            name = str(int(currentframe/fps)) + '.jpg'
            print('Picture saved :::'+ str(int(currentframe/fps)) + '.jpg')
            cv2.imwrite(name, frame) 
        currentframe += 1
    else: 
        break
video.release() 
cv2.destroyAllWindows()