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

- The code for computing the camera matrix and distortion coefficientsin `camera.update_mtx_and_dist`
- in fact, this will be done when the class camera is initialized
- steps to compute the camera matrix and distortion coefficients:
    - compute `objpoints` and `imgpoints`
      - `objpoints` will be appended with a copy of replicated array of coordinates every time I successfully detect all chessboard corners in a test image.
      - `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection `cv2.findChessboardCorners`.  
    - compute camera calibration and distortion coefficients
      - using the `cv2.calibrateCamera` function with `objpoints` and `imgpoints`.
    - apply this distortion correction to the test image using the `cv2.undistort` function and obtained this result.


![alt text][image1]

### Pipeline (single image)

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![alt text][image2]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

- the code to create a thresholded binary image is in `img_camera.combine_threshold`.
- to create a thresholded binary image with the following
    - `abs_sobel_thresh`
    - `mag_thresh`
    - `dir_threshold`
    - `hls_select`
    - `equalize_histogram_color_select`
    - `luv_select`

```
ksize = 3
# Apply each of the thresholding functions
gradx = abs_sobel_thresh(img, orient='x', sobel_kernel=ksize, thresh=(10, 100))
grady = abs_sobel_thresh(img, orient='y', sobel_kernel=ksize, thresh=(10, 100))
mag_binary = mag_thresh(img, sobel_kernel=ksize, thresh=(15, 150))
dir_binary = dir_threshold(img, sobel_kernel=ksize, thresh=(1.0, 1.3))
color_binary = hls_select(img, thresh=(100, 255))
equalize_color_binary = equalize_histogram_color_select(img, thresh=(250, 255))
luv_color = luv_select(img, thresh=(225, 255))

combined = np.zeros_like(img[:,:,0])
combined[((gradx == 1) & (grady == 1) & (dir_binary == 1)) | ((color_binary == 1) & (equalize_color_binary == 1)) | (luv_color==1)] = 1
```
an example of combined threshold image as follow:

![alt text][image3]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

- the code to performe a perspective transform is in `img_camera.transform_perspective`
- steps to performe a perspective transform:
- update source points and destination points
    - `img_camera.update_src_and_dst`
    - considering the symmetry, tune four parameters to show a almost parallel roads in birds-eye perspective.

```
src_points = np.float32([
                        [int(w/2.15), int(h/1.6)],[w - int(w/2.15), int(h/1.6)],
                        [w - w//7, h], [w//7, h]
                        ])
dst_points = np.float32([
                        [int(w/4), 0], [w - int(w/4), 0],
                        [w - int(w/4), h], [int(w/4), h]
                        ])

```
- update perspective transform matrix and inverse matrix
  - `img_camera.update_M_and_Minv`
  - get perspective transform matrix from `cv2.getPerspectiveTransform(src_points, dst_points)`
  - get perspective transform inverse matrix from `cv2.getPerspectiveTransform(dst_points, src_points)`
- then perform a perspective transform
  - `img_camera.transform_perspective`
  - apply `cv2.warpPerspective` to combined thresh image then get image from birds-eye perspective.

![alt text][image4]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

- the code to identified lane-line pixels and fit their positions with a polynomial is in 'Line.find_lines'
- steps to find lines:
- find base point `Line.update_base_points`.
    - find line with windows search `Line.generate_line_fit_with_windows`
    - create windows to search using base points.
    - find points to consist of the lane line
    - then fit line applying the points with `np.polyfit`
- if line has been found in previous frame (and other condition should be satisfied),find line by `update_line_fit`
    - load old lines coefficients
    - find points near the old lines
    - then fit new lines applying the new points with `np.polyfit`


![alt text][image5]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

- the code to calculate the radius of curvature of the lane is in `Line.curvature`
```
# Calculate the new radii of curvature
left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
```
- the code to calculate the position of the vehicle with respect to center is in `Line.distance_from_center`
```
lane_center = (self.left_fitx[-1] + self.right_fitx[-1]) / 2
image_center = self.img.shape[1] / 2
dist_from_center_in_pixels = abs(image_center - lane_center)
dist_from_center_in_meters = dist_from_center_in_pixels * self.xm_per_pix
```

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

- the code to provide the result is in `Line.display`
- display the ultimate result


![alt text][image6]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./ouput_project_video.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

- color threshold method (especially luv color) will performs well on `project_video.mp4` but fail on `challenge_video.mp4`.Single luv color thresholed binary image method will lead to a threshold binary image full of black in some picture then no line will be found so I have to use combined thresholed which will make it more robust.
