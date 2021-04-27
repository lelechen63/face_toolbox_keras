import cv2
from matplotlib import pyplot as plt
import face_alignment
from models.detector.iris_detector import IrisDetector
import dlib
import numpy as np 
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('/raid/celong/lele/github/idinvert_pytorch/utils/shape_predictor_68_face_landmarks.dat')

fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False)


def resize_image(im, max_size=768):
    if np.max(im.shape) > max_size:
        ratio = max_size / np.max(im.shape)
        print(f"Resize image to ({str(int(im.shape[1]*ratio))}, {str(int(im.shape[0]*ratio))}).")
        return cv2.resize(im, (0,0), fx=ratio, fy=ratio)
    return im

im = cv2.imread("images/test5.jpg")[..., ::-1]
im = resize_image(im) # Resize image to prevent GPU OOM.
h, w, _ = im.shape
idet = IrisDetector()
idet.set_detector(fa)

eye_lms = idet.detect_iris(im)

# Display detection result
plt.figure(figsize=(15,10))
draw = idet.draw_pupil(im, eye_lms[0][0,...]) # draw left eye
draw = idet.draw_pupil(draw, eye_lms[0][1,...]) # draw right eye

blank_image = np.zeros((h,w,3), np.uint8)
lms =   eye_lms[0][0,...].astype(np.int32)[:,::-1]

cv2.fillConvexPoly(blank_image, lms[:8], (0,0,255))
cv2.fillConvexPoly(blank_image, lms[8:16], (255,0,0))

lms =   eye_lms[0][1,...].astype(np.int32)[:,::-1]
cv2.fillConvexPoly(blank_image, lms[:8], (0,0,255))
cv2.fillConvexPoly(blank_image, lms[8:16], (255,0,0))
cv2.imwrite('gg.png', blank_image)

gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
# detect faces in the grayscale image
bbox = detector(gray, 1)[0]

x0 = int(bbox.left())
x1 = int(bbox.right())
y0 = int(bbox.top())
y1 = int(bbox.bottom())

plt.subplot(1,2,1)
plt.imshow(draw)
plt.subplot(1,2,2)
plt.imshow(draw[x0:x1, y0:y1])
plt.savefig('foo.png')
