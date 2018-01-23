import numpy as np
import cv2
import glob
from moviepy.editor import VideoFileClip
from IPython.display import HTML

from camera import *
from line import *

import matplotlib.pyplot as plt
import matplotlib.image as mpimage

if os.path.exists('line_fit.p'):
	os.remove('line_fit.p')

if not os.path.exists('unusual_images'):
	os.mkdir('unusual_images')

if os.path.exists('track_records.csv'):
	os.remove('track_records.csv')
	with open('track_records.csv', 'a') as f:
		f.write('{},{},{},{},{},{},{}\n'.format('left_curverad', 'right_curverad', 'dist_from_center_in_meters', 'lane_line_width', 'img_name', 'leftx_base', 'rightx_base'))

def pipeline(img):
	try:
		l = Line(img, auto=True)
	except Exception as e:
		print (str(e))
		img_name = '-'.join([str(x) for x in time.localtime(time.time())[:5]])
		im = Image.fromarray(img)
		im.save("unusual_images/error_{}.jpg".format(img_name))
		raise
	return(l.result)
input_video = 'project_video.mp4'
# input_video = 'challenge_video.mp4'
# clip = VideoFileClip(input_video).subclip(0,1)
clip = VideoFileClip(input_video)
output_clip = clip.fl_image(pipeline)
output_clip.write_videofile('output_' + input_video, audio=False)