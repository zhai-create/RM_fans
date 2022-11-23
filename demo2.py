import math
import numpy as np
import cv2 as cv
from PIL import ImageFont, ImageDraw, Image
import time

class Detector(object):
	def __init__(self):
		self.center = [0,0] #表示圆形轨迹的圆心
		self.radius = 0 #表示圆形轨迹的半径
		self.roix, self.roiy, self.roiw, self.roih = 0, 0, 0, 0
		self.color = ""
		self.target_ls = []
		self.ct = 0 #记录当前帧数
		self.start = 0

	def add_font(self, img, string, position, font_size):
		# 添加文字
		font = ImageFont.truetype("font/simsun.ttc", font_size)
		img_pil = Image.fromarray(img)
		draw = ImageDraw.Draw(img_pil)
		draw.text(position, string, font=font, fill=(0, 0, 255))
		bk_img = np.array(img_pil)
		return bk_img

	def calculate_distance(self, box):
		# 计算box中前两点之间的距离
		distance = math.sqrt((box[0][0] - box[1][0]) ** 2 + math.sqrt((box[0][1] - box[1][1]) ** 2))
		return distance

	def LeastSquaresCircleFitting(self, points):
		# 用于拟合圆形轨迹
		sumX, sumY, sumX2, sumY2, sumX3, sumY3, sumXY, sumX1Y2, sumX2Y1 = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
		N = len(points)
		for i in range(N):
			x = points[i][0]
			y = points[i][1]
			x2 = x ** 2
			y2 = y ** 2
			x3 = x2 * x
			y3 = y2 * y
			xy = x * y
			x1y2 = x * y2
			x2y1 = x2 * y

			sumX += x
			sumY += y
			sumX2 += x2
			sumY2 += y2
			sumX3 += x3
			sumY3 += y3
			sumXY += xy
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
		self.center[0] = a / (-2)
		self.center[1] = b / (-2)
		self.radius = math.sqrt(a * a + b * b - 4 * c) / 2

	def detect_target(self, img):
		# 用于检测大符击打目标的函数
		target = []
		img_roi = img[int(self.roiy):int(self.roiy) + int(self.roih), int(self.roix):int(self.roix) + int(self.roiw)]
		hsv = cv.cvtColor(np.array(img_roi), cv.COLOR_BGR2HSV)  # 将RGB图转换为灰度图

		if (self.color=="red"):
			lower_red1 = np.array([0, 43, 46])
			lower_red2 = np.array([156, 43, 46])
			upper_red1 = np.array([10, 255, 255])
			upper_red2 = np.array([180, 255, 255])
			mask1 = cv.inRange(hsv, lower_red1, upper_red1)  # 根据第一个阈值范围确定mask1
			mask2 = cv.inRange(hsv, lower_red2, upper_red2)  # 根据第二个阈值范围确定mask2
			mask = cv.add(mask1, mask2)  # 将两个mask合并

		elif (self.color == "blue"):
			lower_blue = np.array([100, 43, 46])
			upper_blue = np.array([124, 255, 255])
			mask = cv.inRange(hsv, lower_blue, upper_blue)

		cv.imshow("mask", mask)
		kernel = cv.getStructuringElement(cv.MORPH_RECT, (5, 5)) #获得矩形形状的结构元素
		closing = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)  # 对图像进行闭运算
		contours, hierarchy = cv.findContours(closing, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)  # 对闭运算后的图像寻找轮廓
		contour_num_ls = [0 for i in range(len(contours))]
		if (hierarchy is None):
			return target
		hierarchy = hierarchy[0]
		for i in range(len(contours)):
			if (hierarchy[i][3] != -1):  # 表示如果当前轮廓有父轮廓，则将对应的父轮廓值+1
				contour_num_ls[hierarchy[i][3]] += 1
		box_ls = []
		box_distance_ls = []
		for i in range(len(contour_num_ls)):
			if (contour_num_ls[i] == 1):
				choice_index = hierarchy[i][2]  # 选取仅有一个子轮廓的父轮廓，并将其子轮廓作为target
				rect = cv.minAreaRect(contours[choice_index])  # 最小外接矩形
				box = np.int0(cv.boxPoints(rect))  # 矩形的四个角点取整
				box_ls.append(box)
				box_distance_ls.append(self.calculate_distance(box))
		if (len(box_ls) != 0):
			#过滤噪声边框
			box_distance_ls = np.array(box_distance_ls)
			max_index = np.argmax(box_distance_ls)
			box = box_ls[max_index]
			for j in range(4):
				box[j][0] += int(self.roix)
				box[j][1] += int(self.roiy)
			for j in range(4):
				cv.line(img, box[j], box[(j + 1) % 4], color=(0, 255, 0), thickness=2)
			target = np.int0((box[0] + box[2]) / 2)
		return target

	def predict(self, time_gap, rotation):
		# 大幅运动预测
		a = (0.78+1.045)/2
		w = (1.884+2)/2
		b = 2.09-a
		position_now = self.target_ls[-1]   #(x2,y2)
		angle_now = math.atan((position_now[1] - self.center[1]) / (position_now[0] - self.center[0]))
		delta_angle = ((-a / w) * (-2) * (math.sin(w * 0)) * (math.sin(0.5*w*time_gap))) + b * (
			time_gap)
		angle_predict = angle_now + delta_angle
		if (rotation=="clock"):
			if (position_now[0]>self.center[0]):
				predict_x = self.center[0] - self.radius * math.cos(angle_predict+math.pi)
				predict_y = self.center[1] -  self.radius * math.sin(angle_predict+math.pi)
			else:
				predict_x = self.center[0] - self.radius * math.cos(angle_predict)
				predict_y = self.center[1] - self.radius * math.sin(angle_predict)
		else:
			if (position_now[0] > self.center[0]):
				predict_x = self.center[0] + self.radius * math.cos(angle_predict + math.pi)
				predict_y = self.center[1] + self.radius * math.sin(angle_predict + math.pi)
			else:
				predict_x = self.center[0] + self.radius * math.cos(angle_predict)
				predict_y = self.center[1] + self.radius * math.sin(angle_predict)
		return predict_x, predict_y


	def draw_circle(self, img):
		if (len(self.target_ls)<5):
			return
		self.LeastSquaresCircleFitting(self.target_ls)
		if (self.center[0] != 0 and self.center[1] != 0 and self.radius != 0):
			cv.circle(img, (int(self.center[0]), int(self.center[1])), 5, (255, 0, 0), thickness=cv.FILLED)
			cv.circle(img, (int(self.center[0]), int(self.center[1])), int(self.radius), (255, 255, 0), 5)


	def get_rotation(self):
		# 判断大符的旋转方向
		if (len(self.target_ls) < 10):
			return ""
		temp_target = self.target_ls[len(self.target_ls) - 10:len(self.target_ls)]
		angle_ls = [(temp_target[i][0] - self.center[0]) / (temp_target[i][1] - self.center[1]) for i in
					range(len(temp_target))]
		grad_angle_num_1 = 0
		for i in range(len(angle_ls) - 1):
			if (angle_ls[i + 1] > angle_ls[i]):
				grad_angle_num_1 += 1
		if (grad_angle_num_1 > len(angle_ls) - 1 - grad_angle_num_1):
			rotation = "anti-clock"
		else:
			rotation = "clock"
		return rotation

	def detect(self, file_name, color, time_gap):
		# 逐帧读取视频并检测
		self.color = color
		cap = cv.VideoCapture(file_name)
		while True:
			success, frame = cap.read()  # 逐帧读入
			if success:
				frame = cv.resize(frame, (800, 600))  # 改变每一帧的大小，使画面适应于电脑屏幕界面
				if (self.ct == 0):
					# 如果当前是第一帧，则执行框选ROI的操作
					roi = cv.selectROI(windowName="fans", img=frame, showCrosshair=True,
									   fromCenter=False)
					self.roix, self.roiy, self.roiw, self.roih = roi
					cv.waitKey(-1)
				#绘制ROI矩形
				cv.rectangle(img=frame, pt1=(self.roix-2, self.roiy-2), pt2=(self.roix-2 + self.roiw+2, self.roiy-2 + self.roih+2), color=(0, 0, 255),
							 thickness=2)

				target = self.detect_target(frame) #返回检测到的击打点位置
				if (len(target)!=0):
					self.target_ls.append(target)
					cv.circle(frame, (int(target[0]), int(target[1])), 5, (0, 0, 255), thickness=cv.FILLED)
					print("打", target)


				self.draw_circle(frame)
				rotation = self.get_rotation()
				frame = self.add_font(frame, "rotation:" + rotation, (0, 30), font_size=30)

				if (len(self.target_ls)>10):
					predict_x, predict_y = self.predict(time_gap, "clock")
					cv.circle(frame, (int(predict_x), int(predict_y)), 5, (0, 0, 255), thickness=cv.FILLED)
					print("打——预测",[predict_x,predict_y])
				# if (len(self.target_ls)>10):
				# 	self.predict(start, end)
				# fps = int(1 / (end - start))
				fps = cap.get(cv.CAP_PROP_FPS)
				frame = self.add_font(frame, "FPS:" + str(int(fps)), (0, 0), font_size=30)  # 为当前帧添加帧率
				cv.namedWindow('fans', 0)
				cv.imshow('fans', frame)
				key = cv.waitKey(1) & 0xFF
				if key == 27:  # print('手动停止')
					break
				self.ct += 1
			else:
				print(file_name)
				print('播放完成')
				break
		cap.release()
		cv.destroyAllWindows()


if __name__=="__main__":
	Detector()
	Detector().detect(file_name="red_test1.mp4", color="red", time_gap=0.5)

