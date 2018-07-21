import cv2
import sys

for file_name in sys.argv[1:]:
	img = cv2.imread(file_name)
	gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	cv2.imwrite(file_name, gray)