## Advanced Lane Finding
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)


The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./examples/undistort_output.png "Undistorted"
[image2]: ./output_images/undistort.jpg "Undistorted"
[image3]: ./output_images/combine_threshold.jpg "Binary Example"
[image4]: ./output_images/binary_top_down_image.jpg "binary_top_down_image"
[image5]: ./output_images/find_lane_line.jpg "find_lane_line"
[image6]: ./output_images/result.jpg "Output"
[video1]: ./ouput_project_video.mp4 "Video"


### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Main fies
- camera.py
  - contain two class: camera and img_camera
- line.py
  - contain one class: Line
- image_process.py
  - contain some function for image processing.
- pipeline.py
  - contain pipeline function
  - run `python pipeline.py`, then you will get the output video.
- README.md
- P4.ipynb (can be ignored)
  - prepare image for README step by step
- output_project_video.mp4
- output_images

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

- compute the camera matrix and distortion coefficients
  - camera.update_mtx_and_dist
  - in fact, this will be done when the class camera is initialized

![alt text][image1]

### Pipeline (single image)

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![alt text][image2]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

- create a thresholded binary image
  - img_camera.combine_threshold

```
gradx = abs_sobel_thresh(img, orient='x', sobel_kernel=ksize, thresh=(10, 100))
grady = abs_sobel_thresh(img, orient='y', sobel_kernel=ksize, thresh=(10, 100))
mag_binary = mag_thresh(img, sobel_kernel=ksize, thresh=(15, 150))
dir_binary = dir_threshold(img, sobel_kernel=ksize, thresh=(0.8, 1.3))
color_binary = hls_select(img, thresh=(230, 255))

combined = np.zeros_like(img[:,:,0])
combined[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1)) & color_binary == 1] = 1
```

![alt text][image3]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.
steps:
- update source points and destination points
  - img_camera.update_src_and_dst
- update perspective transform matrix and inverse matrix
  - img_camera.update_M_and_Minv
- then perform a perspective transform
  - img_camera.transform_perspective

![alt text][image4]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

- find lane line from binary top-down image
  - Line.find_lines

![alt text][image5]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

- calculate the radius of curvature of the lane
  - Line.curvature
- calculate the distance from center
  - Line.distance_from_center

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

- display the ultimate result
  - Line.display
![alt text][image6]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./ouput_project_video.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?


We can try to modify the detect result of current frame base on previous result because all kinds of attributions of lane line should be contiuous.
