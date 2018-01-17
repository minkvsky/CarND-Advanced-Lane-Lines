import numpy as np
import cv2
import glob
from moviepy.editor import VideoFileClip
from IPython.display import HTML

from camera import *
from line import *

import matplotlib.pyplot as plt
import matplotlib.image as mpimage

def pipeline(img):
    l = Line(img, auto=True)
    return(l.result)
input_video = 'project_video.mp4'
# clip = VideoFileClip(input_video).subclip(0,1)
clip = VideoFileClip(input_video)
output_clip = clip.fl_image(pipeline)
output_clip.write_videofile('ouput_' + input_video, audio=False)