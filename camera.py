import numpy as np
import cv2
import glob
# import cPickle as pickle
import pickle
import os
import time

# camera class
from image_process import *
class camera():
	# only init is used
	def __init__(self):
		self.img_path = glob.glob('camera_cal/calibration*.jpg') # given a set of chessborad images
		self.img_pattern=(9, 6)
		self.img_size = None
		objp = np.zeros((6*9,3), np.float32)
		objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)

		self.objp = objp
		self.objpoints = [] # 3D
		self.imgpoints = [] # 2D
		self.mtx = None # calibration matrix
		self.dist = None # distortion coefficients
		self.check = False
		# necessary to output?
		if os.path.exists("calibration.p"):
		    calibration_data = pickle.load(open( "calibration.p", "rb" ))
		    self.mtx, self.dist = calibration_data['mtx'], calibration_data['dist']
		    self.check = True
		else:
		    self.update_mtx_and_dist()

	def update_points(self):
		images = [cv2.imread(image) for image in self.img_path]
		for img in images:
			gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
			# update img_size
			self.img_size = gray.shape[::-1]
			ret, corners = cv2.findChessboardCorners(gray, self.img_pattern, None)
			if ret:
				self.objpoints.append(self.objp)
				self.imgpoints.append(corners)

		return self.objpoints, self.imgpoints


	def update_mtx_and_dist(self):
		if self.check:
			return None
		objpoints, imgpoints = self.update_points()
		ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, self.img_size, None, None)
		self.mtx = mtx
		self.dist = dist
		calibration_data = {'mtx':mtx,'dist':dist}
		pickle.dump(calibration_data, open( "calibration.p", "wb" ) )
		self.check = True
		return mtx, dist

class img_camera(camera):
	# update_M_and_Minv and combined_thresh are main
	def __init__(self, img):
		camera.__init__(self)
		self.img = img
		self.img_name = '-'.join([str(x) for x in time.localtime(time.time())[:5]])

		# only relate to img.shape so consistent for a video
		self.src = None
		self.dst = None
		self.M = None
		self.Minv = None

		self.undist = None
		self.combined_threshold_img = None
		self.binary_top_down_image = None
	# need to modify
	def update_src_and_dst(self):
		# only relate to img.shape,is this ok?
		# or make it hard
		# will be tuned
		h, w = self.img.shape[0], self.img.shape[1]

		sx1 = int(np.round(w / 2.15))
		sx2 = w - sx1
		sx4 = w // 7
		sx3 = w - sx4
		sy1 = sy2 = int(np.round(h / 1.6))
		sy3 = sy4 = h

		dx1 = dx4 = int(np.round(w / 4))
		dx2 = dx3 = w - dx1
		dy1 = dy2 = 0
		dy3 = dy4 = h

		src_points = np.float32([[sx1, sy1],[sx2, sy2], [sx3, sy3], [sx4, sy4]])
		dst_points = np.float32([[dx1, dy1], [dx2, dy2], [dx3, dy3], [dx4, dy4]])
		self.src = src_points
		self.dst = dst_points

		return src_points, dst_points

	def update_M_and_Minv(self):
		src_points, dst_points = self.update_src_and_dst()
		self.M = cv2.getPerspectiveTransform(src_points, dst_points)
		self.Minv = cv2.getPerspectiveTransform(dst_points, src_points)
		return self.M, self.Minv

	def transform_perspective(self):
		img_size = self.img.shape[1], self.img.shape[0]
		if self.combined_threshold_img is None:	
			self.combined_thresh()
		warped = cv2.warpPerspective(self.combined_threshold_img, self.M, img_size, flags=cv2.INTER_LINEAR)
		self.binary_top_down_image = warped
		return warped

	def undistort(self):
		undist = cv2.undistort(self.img, self.mtx, self.dist, None, self.mtx)
		self.undist = undist
		return undist


	def combined_thresh(self):
		self.undistort()
		img = self.undist
		# Choose a Sobel kernel size
		ksize = 3
		# Apply each of the thresholding functions
		gradx = abs_sobel_thresh(img, orient='x', sobel_kernel=ksize, thresh=(10, 100))
		grady = abs_sobel_thresh(img, orient='y', sobel_kernel=ksize, thresh=(10, 100))
		mag_binary = mag_thresh(img, sobel_kernel=ksize, thresh=(15, 150))
		dir_binary = dir_threshold(img, sobel_kernel=ksize, thresh=(0.8, 1.3))
		color_binary = hls_select(img, thresh=(230, 255))

		combined = np.zeros_like(img[:,:,0])
		combined[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1)) & color_binary == 1] = 1

		self.combined_threshold_img = combined
		return combined


	

