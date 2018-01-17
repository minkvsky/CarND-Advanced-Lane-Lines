# Define a class to receive the characteristics of each line detection
from camera import *
from PIL import Image
import time

class Line(img_camera):
    def __init__(self, img, auto=False):
        img_camera.__init__(self, img)
        self.result = None
        self.left_fit = None
        self.right_fit = None
        # out put img from find line
        self.out_img = None
        # where should they update
        self.left_fitx = None
        self.right_fitx = None
        self.left_lane_inds = None
        self.right_lane_inds = None

        self.leftx_base = None
        self.rightx_base = None
        self.midpoint = None
        self.lane_line_width = None

        self.ploty = None

        self.leftx = None
        self.lefty = None
        self.rightx = None
        self.righty = None



        # maybe need to correct
        self.ym_per_pix = 30/720 # meters per pixel in y dimension
        self.xm_per_pix = 3.7/700 # meters per pixel in x dimension
        # curvature and vehicle postion in meters
        self.left_curverad = None
        self.right_curverad = None
        self.dist_from_center_in_meters = None

        if self.left_fit is None and os.path.exists('line_fit.p'):
            self.load_line_fit()

        # status process
        # auto update step by step
        # pipeline
        if self.result is None and auto:
            self.update_M_and_Minv()
            self.transform_perspective()
            self.find_lines()
            self.curvature()
            self.distance_from_center()
            self.display()
            if self.dist_from_center_in_meters >= 0.4:
                self.unusual_save()
        

    def update_base_points(self):
        binary_warped = self.binary_top_down_image
        # Assuming you have created a warped binary image called "binary_warped"
        # Take a histogram of the bottom half of the image
        histogram = np.sum(binary_warped[int(binary_warped.shape[0]/2):,:], axis=0)
        # Create an output image to draw on and  visualize the result
        # out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
        # Find the peak of the left and right halves of the histogram
        # These will be the starting point for the left and right lines
        midpoint = np.int(histogram.shape[0]/2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint
        self.leftx_base = leftx_base
        self.rightx_base = rightx_base
        self.midpoint = midpoint
        

        return leftx_base, rightx_base

    def find_lines(self, figname=False):
        # need to be optimized
        binary_warped = self.binary_top_down_image
        self.update_base_points()
        
        if os.path.exists('line_fit.p'):
            left_fit, right_fit = self.update_line_fit()
        else:    
            left_fit, right_fit = self.generate_line_fit_with_windows()

        self.generate_out_img(figname)

        return left_fit, right_fit

    def generate_out_img(self, figname):
        binary_warped = self.binary_top_down_image
        left_fit = self.left_fit
        right_fit = self.right_fit
        left_lane_inds = self.left_lane_inds
        right_lane_inds = self.right_lane_inds
        # Generate x and y values for plotting
        ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
        out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0] # red
        out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255] # blue

        if figname:
            plt.imshow(out_img)
            plt.plot(left_fitx, ploty, color='yellow')
            plt.plot(right_fitx, ploty, color='yellow')
            plt.xlim(0, out_img.shape[1])
            plt.ylim(out_img.shape[0], 0)
        
        self.out_img = out_img
        self.ploty = ploty
        self.left_fitx = left_fitx
        self.right_fitx = right_fitx

        return out_img

    def generate_line_fit_with_windows(self, nwindows = 9, margin = 100, minpix = 50):
            
        binary_warped = self.binary_top_down_image
        leftx_current = self.leftx_base
        rightx_current = self.rightx_base

        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Create empty lists to receive left and right lane pixel indices
        window_height = np.int(binary_warped.shape[0]/nwindows)
        left_lane_inds = []
        right_lane_inds = []

        # Step through the windows one by one
        for window in range(nwindows):
            # Identify window boundaries in x and y (and right and left)
            win_y_low = binary_warped.shape[0] - (window+1)*window_height
            win_y_high = binary_warped.shape[0] - window*window_height
            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin
            # Draw the windows on the visualization image
            cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high), (0,255,0), 2) 
            cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high), (0,255,0), 2) 
            # Identify the nonzero pixels in x and y within the window
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
            # Append these indices to the lists
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)
            # If you found > minpix pixels, recenter next window on their mean position
            if len(good_left_inds) > minpix:
                leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > minpix:        
                rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

        # Concatenate the arrays of indices
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)

        # Extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds] 
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds] 

        # Fit a second order polynomial to each
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)

        self.left_fit = left_fit
        self.right_fit = right_fit
        self.left_lane_inds = left_lane_inds
        self.right_lane_inds = right_lane_inds
        self.lefty = lefty
        self.righty = righty
        self.leftx = leftx
        self.rightx = rightx

        return left_fit, right_fit

    def update_line_fit(self):
        binary_warped = self.binary_top_down_image
        left_fit = self.left_fit
        right_fit = self.right_fit

        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        margin = 100
        left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] - margin)) 
            & (nonzerox < (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] + margin))) 

        right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] - margin)) 
            & (nonzerox < (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] + margin)))  

        # Again, extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds] 
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]
        # Fit a second order polynomial to each
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)

        self.left_fit = left_fit
        self.right_fit = right_fit
        self.left_lane_inds = left_lane_inds
        self.right_lane_inds = right_lane_inds
        self.lefty = lefty
        self.righty = righty
        self.leftx = leftx
        self.rightx = rightx

        return left_fit, right_fit

    def curvature(self, meters=True):
        # default in meters or in pixel
        ploty = self.ploty
        left_fit = self.left_fit
        right_fit = self.right_fit

        # Define y-value where we want radius of curvature
        # I'll choose the maximum y-value, corresponding to the bottom of the image
        y_eval = np.max(ploty)
        # this computation is right?
        if meters:
            # Define conversions in x and y from pixels space to meters
            ym_per_pix = self.ym_per_pix # meters per pixel in y dimension
            xm_per_pix = self.xm_per_pix # meters per pixel in x dimension

            leftx = self.leftx
            lefty = self.lefty
            rightx = self.rightx
            righty = self.righty

            # Fit new polynomials to x,y in world space
            left_fit_cr = np.polyfit(lefty*ym_per_pix, leftx*xm_per_pix, 2)
            right_fit_cr = np.polyfit(righty*ym_per_pix, rightx*xm_per_pix, 2)
            # Calculate the new radii of curvature
            left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
            right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
            # Now our radius of curvature is in meters
            print(left_curverad, 'm', right_curverad, 'm')
            # Example values: 632.1 m    626.2 m
        else:
            left_curverad = ((1 + (2*left_fit[0]*y_eval + left_fit[1])**2)**1.5) / np.absolute(2*left_fit[0])
            right_curverad = ((1 + (2*right_fit[0]*y_eval + right_fit[1])**2)**1.5) / np.absolute(2*right_fit[0])
            print(left_curverad, right_curverad)
        # Example values: 1926.74 1908.48
        self.left_curverad = left_curverad
        self.right_curverad = right_curverad

        return left_curverad, right_curverad

    def distance_from_center(self):
        lane_center = (self.leftx_base + self.rightx_base) / 2
        image_center = self.img.shape[1] / 2
        dist_from_center_in_pixels = abs(image_center - lane_center)
        dist_from_center_in_meters = dist_from_center_in_pixels * self.xm_per_pix
        self.dist_from_center_in_meters = dist_from_center_in_meters
        self.lane_line_width = (self.rightx_base - self.leftx_base) * self.xm_per_pix
        return dist_from_center_in_meters


    def display(self):

        warped = self.binary_top_down_image
        ploty = self.ploty
        left_fitx = self.left_fitx
        right_fitx = self.right_fitx
        Minv = self.Minv
        img = self.img
        self.distance_from_center()
        # Create an image to draw the lines on
        warp_zero = np.zeros_like(warped).astype(np.uint8)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

        # Recast the x and y points into usable format for cv2.fillPoly()
        pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
        pts = np.hstack((pts_left, pts_right))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

        # Warp the blank back to original image space using inverse perspective matrix (Minv)
        newwarp = cv2.warpPerspective(color_warp, Minv, (warped.shape[1], warped.shape[0])) 
        # Combine the result with the original image
        result = cv2.addWeighted(img, 1, newwarp, 0.3, 0)
        text_dist = 'Distance from lane center: {:.3f}m'.format(self.dist_from_center_in_meters)
        text_curvature = 'Radius of curvature: Left {:.3f}m, Right {:3f}m'.format(self.left_curverad, self.right_curverad)
        text_lane_width = 'Width of lane line: {:.3f}m'.format(self.lane_line_width)
        cv2.putText(result, text_lane_width, (30, 170), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        cv2.putText(result, text_dist, (30, 130), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        cv2.putText(result, text_curvature, (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        self.result = result
        # plt.imshow(result)
        return(result)

    def unusual_save(self):
        start = time.time()
        img_name = '-'.join([str(x) for x in time.localtime(start)[:5]])
        im = Image.fromarray(self.img)
        if not os.path.exists('unusual_images'):
            os.mkdir('unusual_images')                
        im.save("unusual_images/unusual_{}.jpg".format(img_name))
        if not os.path.exists('unusual_lines'):
            os.mkdir('unusual_lines')
        pickle.dump(self, open("unusual_images/line_{}.p".format(img_name), "wb" ) )


    def save_line_fit(self):
        line_fit = {'left': self.left_fit, 'right': self.right_fit}
        pickle.dump(line_fit, open('line_fit.p', 'wb'))

    def load_line_fit(self):
        line_fit = pickle.load(open('line_fit.p', 'rb'))
        self.left_fit = line_fit['left']
        self.right_fit = line_fit['right']

