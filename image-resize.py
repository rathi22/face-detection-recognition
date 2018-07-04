from PIL import Image

for i in range(1,41):
	for j in range(1,11):
		img = Image.open("./orl_faces/s"+str(i)+"/"+str(j)+".pgm")
		img.resize((100,100)).save("./orl_faces/s"+str(i)+"/"+str(j)+".png")