import cv2

fC = cv2.CascadeClassifier('haarcascade_frontalface_default.xml') 
sC = cv2.CascadeClassifier('haarcascade_smile.xml') 

def detect(gray, detectSmile): 
	faces = fC.detectMultiScale(gray, 1.3, 5) 
	for (x, y, w, h) in faces: 
		cv2.rectangle(detectSmile, (x, y), ((x + w), (y + h)), (50, 50, 0), 2) 
		roiG = gray[y:y + h, x:x + w] 
		roiC = detectSmile[y:y + h, x:x + w] 
		smile = sC.detectMultiScale(roiG, 1.8, 20) 

		for (sx, sy, sw, sh) in smile: 
			cv2.rectangle(roiC, (sx, sy), ((sx + sw), (sy + sh)), (255, 0, 100), 2) 
	return detectSmile 

v = cv2.VideoCapture(0) 
while True: 
	_,detectSmile = v.read() 			 
	gray = cv2.cvtColor(detectSmile, cv2.COLOR_BGR2GRAY) 
	mySmile = detect(gray, detectSmile) 			 
	cv2.imshow('Video', mySmile); cv2.imwrite('MySmile.png', mySmile) 		 
	if cv2.waitKey(1) & 0xff == ord('q'):			 
		break
v.release()								 
cv2.destroyAllWindows() 
