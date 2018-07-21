import sys
import torch
import cv2 
from PIL import Image
import torchvision

model = torch.load(sys.argv[1])
image_file = sys.argv[2]
face_cascade = cv2.CascadeClassifier("Celebs/haarcascade_frontalface_default.xml")

img = cv2.imread(image_file)
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray,1.3,5)

index=1
for (x,y,w,h) in faces:
	face = img[y:y+h, x:x+w]
	file_name = "temp/%d.png"%index
	cv2.imwrite(file_name, face)
	face = Image.open(file_name)
	face.resize((100,100)).save(file_name)
	trans = torchvision.transforms.ToTensor()
	face = Image.open(file_name)
	tensor = trans(face)
	shape = tensor.size()
	tensor = tensor.view(1,3,100,100)
	output = model(tensor)
	output = output.data.max(1, keepdim=True)[1]
	print(output.item()+1)
	index += 1
	cv2.putText(img,"%d"%output.item(), (x,y+h+15), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255),1)
	# img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
# cv2.imwrite("_"+image_file, img)


# file_name = file_num+".png"
# cv2.imwrite(file_name, img)
# img = Image.open(file_name)
# img.resize((100,100)).save(file_name)