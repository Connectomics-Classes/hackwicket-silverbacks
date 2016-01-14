import os
from os import listdir
from os.path import isfile, join
from PIL import Image

#put files in current directory into list
#files=["mito_anno_000041.png"]
files = [f for f in listdir('.') if f.endswith(".png")]
current_color = (255,255,255)

for f in files:
	picture = Image.open(f)
	picture = picture.convert('RGB')
	pixels = picture.load()
		#image is grayscale with 16 bits

	for x in range(picture.size[0]):
		for y in range(picture.size[1]):
			if pixels[x,y] == (0,0,0):
				pixels[x,y] = current_color
			else:
				pixels[x,y] = (0,0,200) 
	
	#picture.show()
	#picture.save("/home/eric/Documents/hackwicket-silverbacks/hackwicket-silverbacks/data/improved_annotations/improved_"+str(f), "png")
	
	#turn all current_color pixels transparent
	transparent_pic = picture.convert("RGBA")
	datas = transparent_pic.getdata()
	
	newData = []
	for item in datas:
		if item[0] == 255 and item[1] == 255 and item[2] == 255:
			newData.append((255, 255, 255, 0))
		else:
			newData.append(item)
	transparent_pic.putdata(newData)
	#transparent_pic.show()
	transparent_pic.save("/home/eric/Documents/hackwicket-silverbacks/hackwicket-silverbacks/data/improved_annotations/improved_"+str(f), "png")
	
	
			
	
			
