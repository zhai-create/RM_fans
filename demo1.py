import math

import numpy as np
import cv2 as cv
import time
from PIL import ImageFont, ImageDraw, Image

def add_font(img, string, position, font_size):
	# 添加文字
	font = ImageFont.truetype("font/simsun.ttc", font_size)
	img_pil = Image.fromarray(img)
	draw = ImageDraw.Draw(img_pil)
	draw.text(position, string, font=font, fill=(0, 0, 255))
	bk_img = np.array(img_pil)
	return bk_img

def calculate_distance(box):
	# 计算box中前两点之间的距离
	distance = math.sqrt((box[0][0]-box[1][0])**2+math.sqrt((box[0][1]-box[1][1])**2))
	return distance

def LeastSquaresCircleFitting(points):
	center = [0, 0]
	sumX, sumY, sumX2, sumY2, sumX3, sumY3, sumXY, sumX1Y2, sumX2Y1 = 0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0
	N = len(points)
	for i in range(N):
		x = points[i][0]
		y = points[i][1]
		x2 = x**2
		y2 = y**2
		x3 = x2 * x
		y3 = y2 * y
		xy = x*y
		x1y2 = x * y2
		x2y1 = x2 * y

		sumX += x
		sumY += y
		sumX2 += x2
		sumY2 += y2
		sumX3 += x3
		sumY3 += y3
		sumXY +=xy
		sumX1Y2 += x1y2
		sumX2Y1 += x2y1

	C = N * sumX2 - sumX * sumX
	D = N * sumXY - sumX * sumY
	E = N * sumX3 + N * sumX1Y2 - (sumX2 + sumY2) * sumX
	G = N * sumY2 - sumY * sumY
	H = N * sumX2Y1 + N * sumY3 - (sumX2 + sumY2) * sumY
	denominator = C * G - D * D
	if denominator == 0:
		return [0, 0], 0
	a = (H * D - E * G) / (denominator)
	denominator = D * D - G * C
	b = (H * C - E * D) / (denominator)
	c = -(a * sumX + b * sumY + sumX2 + sumY2) / N
	center[0] = a / (-2)
	center[1] = b / (-2)
	r = math.sqrt(a * a + b * b - 4 * c) / 2
	return center, r



def detect_target(img, roix, roiy, roiw, roih, color):
	target = []
	roix += 2
	roiy += 2
	roiw -= 2
	roih -= 2
	img_roi = img[int(roiy):int(roiy)+int(roih), int(roix):int(roix)+int(roiw)]
	hsv = cv.cvtColor(np.array(img_roi), cv.COLOR_BGR2HSV)  # 将RGB图转换为灰度图

	if (color=="red"):
		lower_red1 = np.array([0, 43, 46])
		lower_red2 = np.array([156, 43, 46])
		upper_red1 = np.array([10, 255, 255])
		upper_red2 = np.array([180, 255, 255])
		mask1 = cv.inRange(hsv, lower_red1, upper_red1)  #根据第一个阈值范围确定mask1
		mask2 = cv.inRange(hsv, lower_red2, upper_red2)  #根据第二个阈值范围确定mask2
		mask = cv.add(mask1, mask2) #将两个mask合并
	elif (color=="blue"):
		lower_blue = np.array([100, 43, 46])
		upper_blue = np.array([124, 255, 255])
		mask = cv.inRange(hsv, lower_blue, upper_blue)

	cv.imshow("mask", mask)
	kernel = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))
	closing = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel) #对图像进行闭运算
	contours, hierarchy = cv.findContours(closing, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE) #对闭运算后的图像寻找轮廓
	contour_num = [0 for i in range(len(contours))]
	if (hierarchy is None):
		return target
	hierarchy = hierarchy[0]
	for i in range(len(contours)):
		if (hierarchy[i][3]!=-1): #表示如果当前轮廓有父轮廓，则将对应的父轮廓值+1
			contour_num[hierarchy[i][3]]+=1
	box_ls = []
	box_distance_ls = []
	for i in range(len(contour_num)):
		if (contour_num[i]==1):
			choice_index = hierarchy[i][2] #选取仅有一个子轮廓的父轮廓，并将其子轮廓作为target
			rect = cv.minAreaRect(contours[choice_index])  # 最小外接矩形
			box = np.int0(cv.boxPoints(rect))  # 矩形的四个角点取整
			box_ls.append(box)
			box_distance_ls.append(calculate_distance(box))
	if (len(box_ls)!=0):
		box_distance_ls  = np.array(box_distance_ls)
		max_index = np.argmax(box_distance_ls)
		box = box_ls[max_index]
		for j in range(4):
			box[j][0] += int(roix)
			box[j][1] += int(roiy)
		for j in range(4):
			cv.line(img, box[j], box[(j + 1) % 4], color=(0, 255, 0), thickness=2)
		target = np.int0((box[0] + box[2]) / 2)
	return target

def draw_circle(img, target_ls):
	center = [0,0]
	if (len(target_ls)<5):
		return center
	else:
		# if (len(target_ls)>100):
		# 	target_ls.pop(0)
		center, radius = LeastSquaresCircleFitting(target_ls)
		if (center[0]!=0 and center[1]!=0 and radius!=0):
			cv.circle(img, (int(center[0]) , int(center[1])), 5, (255, 0, 0), thickness=cv.FILLED)
			cv.circle(img, (int(center[0]), int(center[1])), int(radius), (255, 255, 0),5)
			return center
		return center

def get_rotation(center, target_ls):
	if (len(target_ls)<10):
		return ""
	temp_target = target_ls[len(target_ls)-10:len(target_ls)]
	angle_ls = [(temp_target[i][0]-center[0])/(temp_target[i][1]-center[1])  for i in range(len(temp_target))]
	grad_angle_num_1 = 0
	for i in range(len(angle_ls)-1):
		if (angle_ls[i+1]>angle_ls[i]):
			grad_angle_num_1 += 1
	if (grad_angle_num_1>len(angle_ls)-1-grad_angle_num_1):
		rotation = "anti-block"
	else:
		rotation = "block"
	return rotation



def detect(file_name):
	ct = 0 #记录当前的帧数
	cap = cv.VideoCapture(file_name)
	ROI = (0,0,0,0)
	target_ls = []
	while True:
		success, frame = cap.read()  # 逐帧读入
		if success:
			frame = cv.resize(frame, (800, 600))  # 改变每一帧的大小，使画面适应于电脑屏幕界面
			if ct == 0:
				# 如果当前是第一帧，则执行框选ROI的操作
				roi = cv.selectROI(windowName="fans", img=frame, showCrosshair=True,
								   fromCenter=False)
				ROI = roi
				cv.waitKey(-1)
			roix, roiy, roiw, roih = ROI
			# 绘制ROI矩形
			cv.rectangle(img=frame, pt1=(roix, roiy), pt2=(roix + roiw, roiy + roih), color=(0, 0, 255), thickness=2)
			target = detect_target(frame, roix, roiy, roiw, roih, color="red")
			if (len(target)!=0):
				target_ls.append(target)
				cv.circle(frame, (int(target[0]), int(target[1])), 5, (0, 0, 255), thickness=cv.FILLED)
			center = draw_circle(frame, target_ls)
			rotation = get_rotation(center, target_ls)
			frame = add_font(frame, "rotation:" + rotation, (0, 30), font_size=30)
			# get_rotation()


			fps = cap.get(cv.CAP_PROP_FPS)
			frame = add_font(frame, "FPS:" + str(int(fps)), (0, 0), font_size=30)  # 为当前帧添加帧率
			cv.namedWindow('fans', 0)
			cv.imshow('fans', frame)
			# key = cv.waitKey(int(1000/25)) & 0xFF
			key = cv.waitKey(2) & 0xFF
			# key = cv.waitKey(5)
			if key == 27: #				print('手动停止')
				break
			ct += 1
		else:
			print(file_name)
			print('播放完成')
			break
	cap.release()
	cv.destroyAllWindows()

if __name__=="__main__":
	detect("red_test1.mp4")














# kernel = np.ones((1, 5), np.uint8)
# img = cv2.imread("test.jpeg")
# gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# ret, binary = cv2.threshold(gray,127,255,cv2.THRESH_BINARY)
# binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, anchor=(2, 0), iterations=5)
# contours, hierarchy = cv2.findContours(binary,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
# print("contours", len(contours))
#
# print("hierarchy", hierarchy)
# for i in range(len(contours)):
# 	print("轮廓{}:".format(i))
# 	print(contours[i].shape)
# cv2.drawContours(img,contours,-1,(0,0,255),3)
# cv2.imshow("img", img)
# cv2.waitKey(0)
