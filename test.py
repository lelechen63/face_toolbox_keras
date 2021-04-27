import cv2
from matplotlib import pyplot as plt
from models.detector import face_detector
from models.detector.iris_detector import IrisDetector
import dlib
import numpy as np 


# detector = dlib.get_frontal_face_detector()
# predictor = dlib.shape_predictor('/raid/celong/lele/github/idinvert_pytorch/utils/shape_predictor_68_face_landmarks.dat')

fd = face_detector.FaceAlignmentDetector(
    lmd_weights_path="./models/detector/FAN/2DFAN-4_keras.h5"# 2DFAN-4_keras.h5, 2DFAN-1_keras.h5
)

def resize_image(im, max_size=768):
    if np.max(im.shape) > max_size:
        ratio = max_size / np.max(im.shape)
        print(f"Resize image to ({str(int(im.shape[1]*ratio))}, {str(int(im.shape[0]*ratio))}).")
        return cv2.resize(im, (0,0), fx=ratio, fy=ratio)
    return im

im = cv2.imread("images/test5.jpg")[..., ::-1]
im = resize_image(im) # Resize image to prevent GPU OOM.
h, w, _ = im.shape
# plt.imshow(im)
idet = IrisDetector()
idet.set_detector(fd)

eye_lms = idet.detect_iris(im)
# print (eye_lms.shape)
# Display detection result
plt.figure(figsize=(15,10))
draw = idet.draw_pupil(im, eye_lms[0][0,...]) # draw left eye
draw = idet.draw_pupil(draw, eye_lms[0][1,...]) # draw right eye

bboxes = fd.detect_face(im, with_landmarks=False)
x0, y0, x1, y1, _ = bboxes[0].astype(np.int32)


# gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
# # detect faces in the grayscale image
# bbox = detector(gray, 1)[0]

# x0 = int(bbox.left())
# x1 = int(bbox.right())
# y0 = int(bbox.top())
# y1 = int(bbox.bottom())

plt.subplot(1,2,1)
plt.imshow(draw)
plt.subplot(1,2,2)
plt.imshow(draw[x0:x1, y0:y1])
plt.savefig('foo.png')
