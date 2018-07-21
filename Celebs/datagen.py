from bs4 import BeautifulSoup
import requests
import re
from urllib import request
import os
import argparse
import sys
import json
import shutil
import cv2
from PIL import Image

# adapted from http://stackoverflow.com/questions/20716842/python-download-images-from-google-image-search
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

def is_valid(file_num):
	img = cv2.imread(file_num+".jpg")
	gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	faces = face_cascade.detectMultiScale(gray,1.3,5)
	# print(len(faces))
	if len(faces)!=1:
		return False	
	for (x,y,w,h) in faces[0:1]:
		img = img[y:y+h, x:x+w]
	file_name = file_num+".png"
	cv2.imwrite(file_name, img)
	img = Image.open(file_name)
	img.resize((100,100)).save(file_name)
	return True 

def get_soup(url,header):
	return BeautifulSoup(request.urlopen(request.Request(url,headers=header)),'html.parser')

def main(args):
	parser = argparse.ArgumentParser(description='Scrape Google images')
	parser.add_argument('-s', '--search', default='bananas', type=str, help='search term')
	parser.add_argument('-n', '--num_images', default=10, type=int, help='num images to save')
	parser.add_argument('-d', '--directory', default='./', type=str, help='save directory')
	args = parser.parse_args()
	query = args.search#raw_input(args.search)
	max_images = args.num_images
	save_directory = args.directory
	image_type="Action"
	query= query.split()
	query='+'.join(query)
	print("Searching for %s...\n"%(args.search))
	url="https://www.google.co.in/search?q="+query+"&source=lnms&tbm=isch"
	header={'User-Agent':"Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/43.0.2357.134 Safari/537.36"}
	soup = get_soup(url,header)
	ActualImages=[]# contains the link for Large original images, type of  image
	for a in soup.find_all("div",{"class":"rg_meta"}):
		link , Type =json.loads(a.text)["ou"]  ,json.loads(a.text)["ity"]
		ActualImages.append((link,Type))
	count = 0
	flag = 0
	for i , (img_link , Type) in enumerate( ActualImages[0:60]):
		r = requests.get(img_link,stream=True)
		if r.status_code == 200 and ( Type=="jpg" or Type=="jpeg"):
			file_name = save_directory+str(count+1)+".jpg"
			with open(file_name, 'wb') as f:
				r.raw.decode_content = True
				shutil.copyfileobj(r.raw, f)
			if is_valid(save_directory+str(count+1)):
				count += 1
			os.remove(file_name)
			if count==max_images:
				flag = 1
				break
	if flag==0:
		print("Try some other Celeb\n")

if __name__ == '__main__':
	from sys import argv
	try:
		main(argv)
	except KeyboardInterrupt:
		pass
	sys.exit()
