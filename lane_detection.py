import numpy as np
import pandas as pd
import cv2
import os
import glob
import matplotlib.pyplot as plt
import pickle
import math
import globals 

from alpr_lib import alpr
#from main import frame_rate

# LOCAL VALUES
frame_rate = 30
find_danger_driver = 0
danger_ticktime = 0
danger_candidate_id = ''

# 定義點
class Point:
    def __init__(self,x=0,y=0):
        self.x=x
        self.y=y
    def getx(self):
        return self.x
    def gety(self):
        return self.y 
# get len 
class Getlen:
    def __init__(self,p1,p2):
        self.x=p1.getx()-p2.getx()
        self.y=p1.gety()-p2.gety()
        #用math.sqrt（）求平方根
        self.len= math.sqrt((self.x**2)+(self.y**2))
    def getlen(self):
        return self.len

def cal_focal():
    img = cv2.imread('camera_cal/forcalLength_28.6_105.jpg')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    ret, corners = cv2.findChessboardCorners(gray, (9,6), None)
    if ret == True:
        print('corners ', len(corners))
        
    for index, point in enumerate(corners):
        tmp = (point[0][0], point[0][1])
        cv2.circle(img, tmp, 30, (0, 0, 255), 4)

    # 選取中間, 來計算焦距
    corner = corners[18]
    tmp1 = (corner[0][0], corner[0][1])
    cv2.circle(img, tmp1, 30, (0, 255, 0), 4)

    corner = corners[18+8]
    tmp2 = (corner[0][0], corner[0][1])
    cv2.circle(img, tmp2, 30, (0, 255, 0), 4)


    cv2.imwrite('camera_cal/chessboard_focallength.jpg', img)
     
    p1=Point(tmp1[0], tmp1[1])
    p2=Point(tmp2[0], tmp2[1])
    l=Getlen(p1,p2)
    
    dist = l.getlen()

    focalLength = (dist * 105) / 28.6

    print('focalLength', focalLength)

    pickle.dump(focalLength, open('camera_cal/focallength.p', 'wb'))

def get_focaLength():
    return pickle.load(open('camera_cal/focallength.p', mode='rb'))


import math

def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    you should call plt.imshow(gray, cmap='gray')"""
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def gaussian_noise(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)

def region_of_interest(img, vertices):
    """
    Applies an image mask.
    
    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)   
    
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    #filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def draw_lines(img, lines, color=[255, 0, 0], thickness=10):
    """
    NOTE: this is the function you might want to use as a starting point once you want to 
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).  
    
    Think about things like separating line segments by their 
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of 
    the lines and extrapolate to the top and bottom of the lane.
    
    This function draws `lines` with `color` and `thickness`.    
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """
    top = 320
    bottom = 550
    left_x1s = []
    left_y1s = []
    left_x2s = []
    left_y2s = []
    right_x1s = []
    right_y1s = []
    right_x2s = []
    right_y2s = []
    for line in lines:
        #print(line)
        # Feel this is the brute force method, but I'm time constrained. I will research ideal numpy methods later.
        for x1,y1,x2,y2 in line:
            # Draw line segments in blue for error checking.
            cv2.line(img, (x1, y1), (x2, y2), [0, 0, 255], 6)
            
            slope = get_slope(x1,y1,x2,y2)
            if slope < 0:
                # Ignore obviously invalid lines
                if slope > -.5 or slope < -.8:
                    continue        
                left_x1s.append(x1)
                left_y1s.append(y1)
                left_x2s.append(x2)
                left_y2s.append(y2)
            else:
                # Ignore obviously invalid lines
                if slope < .5 or slope > .8:
                    continue        
                right_x1s.append(x1)
                right_y1s.append(y1)
                right_x2s.append(x2)
                right_y2s.append(y2)
                
    try:
        avg_right_x1 = int(np.mean(right_x1s))
        avg_right_y1 = int(np.mean(right_y1s))
        avg_right_x2 = int(np.mean(right_x2s))
        avg_right_y2 = int(np.mean(right_y2s))
        right_slope = get_slope(avg_right_x1,avg_right_y1,avg_right_x2,avg_right_y2)

        right_y1 = top
        right_x1 = int(avg_right_x1 + (right_y1 - avg_right_y1) / right_slope)
        right_y2 = bottom
        right_x2 = int(avg_right_x1 + (right_y2 - avg_right_y1) / right_slope)
        cv2.line(img, (right_x1, right_y1), (right_x2, right_y2), color, thickness)
    except ValueError:
        # Don't error when a line cannot be drawn
        pass

    try:
        avg_left_x1 = int(np.mean(left_x1s))
        avg_left_y1 = int(np.mean(left_y1s))
        avg_left_x2 = int(np.mean(left_x2s))
        avg_left_y2 = int(np.mean(left_y2s))
        left_slope = get_slope(avg_left_x1,avg_left_y1,avg_left_x2,avg_left_y2)

        left_y1 = top
        left_x1 = int(avg_left_x1 + (left_y1 - avg_left_y1) / left_slope)
        left_y2 = bottom
        left_x2 = int(avg_left_x1 + (left_y2 - avg_left_y1) / left_slope)
        cv2.line(img, (left_x1, left_y1), (left_x2, left_y2), color, thickness)        
    except ValueError:
        # Don't error when a line cannot be drawn
        pass
    
def get_slope(x1,y1,x2,y2):
    return ((y2-y1)/(x2-x1))

def get_x(x,y,dy,slope):
    return
            
def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.
        
    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    #print(lines)
    # Make a RGB shape of the correct dimensions
    shape = (img.shape[0], img.shape[1], 3)
    line_img = np.zeros(shape, dtype=np.uint8)

    draw_lines(line_img, lines)
    return line_img

# Python 3 has support for cool math symbols.

def weighted_img(img, initial_img, α=.8, β=1., λ=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.
    
    `initial_img` should be the image before any processing.
    
    The result image is computed as follows:
    
    initial_img * α + img * β + λ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, λ)


def hough_process_image(image):
    # NOTE: The output you return should be a color image (3 channel) for processing video below
    # TODO: put your pipeline here,
    # you should return the final output (image where lines are drawn on lanes)
    gray_image = grayscale(image)
    gaus_blur = gaussian_noise(gray_image, 3)
    edges = canny(gaus_blur, 50,150)    
    imshape = image.shape

    vertices = np.array([[(0,imshape[0]),(450, 320), (500, 320), (imshape[1],imshape[0])]], dtype=np.int32)
    masked = region_of_interest(edges, vertices)

    rho = 2            #distance resolution in pixels of the Hough grid
    theta = np.pi/180  #angular resolution in radians of the Hough grid
    threshold = 5     #minimum number of votes (intersections in Hough grid cell)
    min_line_len = 10  #minimum number of pixels making up a line
    max_line_gap = 20  #maximum gap in pixels between connectable line segments
    line_image = hough_lines(masked, rho, theta, threshold, min_line_len, max_line_gap)

    result = weighted_img(line_image, image)
    return result

def undistort_img():
    chessboard_size = (9, 6)#(4, 4)
    # Prepare object points 0,0,0 ... 8,5,0
    obj_pts = np.zeros(( chessboard_size[0]*chessboard_size[1] ,3), np.float32)
    obj_pts[:,:2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1,2)

    # Stores all object points & img points from all images
    objpoints = []
    imgpoints = []

    # Get directory for all calibration images
    images = glob.glob('camera_cal/*.jpg')

    for indx, fname in enumerate(images):
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        ret, corners = cv2.findChessboardCorners(gray, (chessboard_size[0], chessboard_size[1]), None)

        if ret == True:
            print('found the chessboard corners')
            objpoints.append(obj_pts)
            imgpoints.append(corners)
    # Test undistortion on img
    img_size = (img.shape[1], img.shape[0])

    # Calibrate camera
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, None,None)

    dst = cv2.undistort(img, mtx, dist, None, mtx)
    # Save camera calibration for later use
    dist_pickle = {}
    dist_pickle['mtx'] = mtx
    dist_pickle['dist'] = dist
    pickle.dump( dist_pickle, open('camera_cal/cal_pickle.p', 'wb') )

def undistort(img, cal_dir='camera_cal/cal_pickle.p'):
    #cv2.imwrite('camera_cal/test_cal.jpg', dst)
    with open(cal_dir, mode='rb') as f:
        file = pickle.load(f)
    mtx = file['mtx']
    dist = file['dist']
    dst = cv2.undistort(img, mtx, dist, None, mtx)

    return dst

def pipeline(img, s_thresh=(100, 255), sx_thresh=(15, 255)):
    img = undistort(img)
    img = np.copy(img)
    # Convert to HLS color space and separate the V channel
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(np.float)
    l_channel = hls[:,:,1]
    s_channel = hls[:,:,2]
    h_channel = hls[:,:,0]
    # Sobel x
    sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 1) # Take the derivative in x

    abs_sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))

    # Threshold x gradient
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1

    # Threshold color channel
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1

    color_binary = np.dstack((np.zeros_like(sxbinary), sxbinary, s_binary)) * 255

    combined_binary = np.zeros_like(sxbinary)
    combined_binary[(s_binary == 1) | (sxbinary == 1)] = 1
    
    return combined_binary

def perspective_warp(ori_img,
                     img,
                     dst_size=(1280,720),
                     src=np.float32([(0.37,0.75),(0.58,0.75),(0.1,1),(1,1)]),
                     #src=np.float32([(0.43,0.65),(0.58,0.65),(0.1,1),(1,1)]),
                     #src=np.float32([(0.25,0.25),(0.75,0.25),(0.25,1),(0.75,1)]),
                     dst=np.float32([(0,0), (1, 0), (0,1), (1,1)])):
    img_size = np.float32([(img.shape[1],img.shape[0])])
    src = src* img_size

    # For destination points, I'm arbitrarily choosing some points to be
    # a nice fit for displaying our warped result 
    # again, not exact, but close enough for our purposes
    dst = dst * np.float32(dst_size)

    # Given src and dst points, calculate the perspective transform matrix
    M = cv2.getPerspectiveTransform(src, dst)
    # Warp the image using OpenCV warpPerspective()
    warped = cv2.warpPerspective(img, M, dst_size)


    # Draw safe zoone
    print(src)
    p0 = [int(src[0][0][0]), int(src[0][0][1])]
    p1 = [int(src[0][1][0]), int(src[0][1][1])]
    p2 = [int(src[0][2][0]), int(src[0][2][1])]
    p3 = [int(src[0][3][0]), int(src[0][3][1])]

    pts=np.array([p0, p1, p3, p2],np.int32)
    pts=pts.reshape((-1,1,2))
    output = ori_img.copy()
    cv2.polylines(output,[pts],True,(255,0,0),10)

    return warped, output

def inv_perspective_warp(img, 
                     dst_size=(1280,720),
                     src=np.float32([(0,0), (1, 0), (0,1), (1,1)]),
                     dst=np.float32([(0.37,0.75),(0.58,0.75),(0.1,1),(1,1)])):
                     #dst=np.float32([(0.43,0.65),(0.58,0.65),(0.1,1),(1,1)])):
    img_size = np.float32([(img.shape[1],img.shape[0])])
    src = src* img_size
    # For destination points, I'm arbitrarily choosing some points to be
    # a nice fit for displaying our warped result 
    # again, not exact, but close enough for our purposes
    dst = dst * np.float32(dst_size)
    # Given src and dst points, calculate the perspective transform matrix
    M = cv2.getPerspectiveTransform(src, dst)
    # Warp the image using OpenCV warpPerspective()
    warped = cv2.warpPerspective(img, M, dst_size)
    return warped

def get_hist(img):
    hist = np.sum(img[img.shape[0]//2:,:], axis=0)
    return hist



left_a, left_b, left_c = [],[],[]
right_a, right_b, right_c = [],[],[]

def sliding_window(img, nwindows=9, margin=150, minpix = 1, draw_windows=True):
    global left_a, left_b, left_c,right_a, right_b, right_c
    left_fit_= np.empty(3)
    right_fit_ = np.empty(3)
    out_img = np.dstack((img, img, img))*255

    histogram = get_hist(img)
    # find peaks of left and right halves
    midpoint = int(histogram.shape[0]/2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # Set height of windows
    window_height = np.int(img.shape[0]/nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = img.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base

    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = img.shape[0] - (window+1)*window_height
        win_y_high = img.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        # Draw the windows on the visualization image
        if draw_windows == True:
            cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),
            (100,255,255), 3)
            cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),
            (100,255,255), 3)
        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
        (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
        (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:        
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))
        
        
#        if len(good_right_inds) > minpix:        
#            rightx_current = np.int(np.mean([leftx_current +900, np.mean(nonzerox[good_right_inds])]))
#        elif len(good_left_inds) > minpix:
#            rightx_current = np.int(np.mean([np.mean(nonzerox[good_left_inds]) +900, rightx_current]))
#        if len(good_left_inds) > minpix:
#            leftx_current = np.int(np.mean([rightx_current -900, np.mean(nonzerox[good_left_inds])]))
#        elif len(good_right_inds) > minpix:
#            leftx_current = np.int(np.mean([np.mean(nonzerox[good_right_inds]) -900, leftx_current]))


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
    
    left_a.append(left_fit[0])
    left_b.append(left_fit[1])
    left_c.append(left_fit[2])
    
    right_a.append(right_fit[0])
    right_b.append(right_fit[1])
    right_c.append(right_fit[2])
    
    left_fit_[0] = np.mean(left_a[-10:])
    left_fit_[1] = np.mean(left_b[-10:])
    left_fit_[2] = np.mean(left_c[-10:])
    
    right_fit_[0] = np.mean(right_a[-10:])
    right_fit_[1] = np.mean(right_b[-10:])
    right_fit_[2] = np.mean(right_c[-10:])
    
    # Generate x and y values for plotting
    ploty = np.linspace(0, img.shape[0]-1, img.shape[0] )
    left_fitx = left_fit_[0]*ploty**2 + left_fit_[1]*ploty + left_fit_[2]
    right_fitx = right_fit_[0]*ploty**2 + right_fit_[1]*ploty + right_fit_[2]

    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 100]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 100, 255]
    
    return out_img, (left_fitx, right_fitx), (left_fit_, right_fit_), ploty

def get_curve(img, leftx, rightx):

    width = img.shape[0]
    length = img.shape[1]

    #print('shape {0}:{1}'.format(img.shape[0], img.shape[1]))
    ploty = np.linspace(0, img.shape[0]-1, img.shape[0])
    y_eval = np.max(ploty)
    #ym_per_pix = 30.5/720 # meters per pixel in y dimension
    #xm_per_pix = 3.7/720 # meters per pixel in x dimension

    ym_per_pix = 30.5/length# meters per pixel in y dimension
    xm_per_pix = 3.7/width # meters per pixel in x dimension


    # Fit new polynomials to x,y in world space
    left_fit_cr = np.polyfit(ploty*ym_per_pix, leftx*xm_per_pix, 2)
    right_fit_cr = np.polyfit(ploty*ym_per_pix, rightx*xm_per_pix, 2)
    # Calculate the new radii of curvature
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])

    car_pos = img.shape[1]/2
    l_fit_x_int = left_fit_cr[0]*img.shape[0]**2 + left_fit_cr[1]*img.shape[0] + left_fit_cr[2]
    r_fit_x_int = right_fit_cr[0]*img.shape[0]**2 + right_fit_cr[1]*img.shape[0] + right_fit_cr[2]
    lane_center_position = (r_fit_x_int + l_fit_x_int) /2
    center = (car_pos - lane_center_position) * xm_per_pix / 10
    # Now our radius of curvature is in meters
    return (left_curverad, right_curverad, center)

def draw_lanes(img, left_fit, right_fit, perspective_ratio = np.float32([(0.37,0.75),(0.58,0.75),(0.1,1),(1,1)])):

    width = img.shape[0]
    length = img.shape[1]

    ploty = np.linspace(0, img.shape[0]-1, img.shape[0])
    color_img = np.zeros_like(img)
    
    left = np.array([np.transpose(np.vstack([left_fit, ploty]))])
    right = np.array([np.flipud(np.transpose(np.vstack([right_fit, ploty])))])

    #print("left ", left)
    #print("right ", right)


    points = np.hstack((left, right))
    
    #cv2.polylines(color_img,[points],True,(0,200,255),5)
    cv2.fillPoly(color_img, np.int_(points), (0,200,255))
    inv_perspective = inv_perspective_warp(color_img, dst_size=(length,width), dst = perspective_ratio)
    inv_perspective = cv2.addWeighted(img, 1, inv_perspective, 0.7, 0)
    return inv_perspective

def draw_safe_zone(img, carsPos, lp_strs, kmh):
    global find_danger_driver
    global danger_ticktime
    global danger_candidate_id
    car_detected_list = []

    safe_zone_color = (0, 255, 0)

    overlay = img.copy()
    output = img.copy()

    safe_zone_mask = np.zeros((output.shape[0], output.shape[1], 1), np.uint8)

    alpha = 0.8

    img_size = np.float32([(img.shape[1],img.shape[0])])
    src=np.float32([(0.35,0.85),(0.60,0.85),(0.20,1),(0.75,1)]),
    src = src* img_size

    # Draw safe zone
    p0 = src[0][0]
    p1 = src[0][1]
    p2 = src[0][2]
    p3 = src[0][3]
    
    # save safe zone for training data
    globals.safe_zone_p0_x = p0[0]
    globals.safe_zone_p0_y = p0[1]
    globals.safe_zone_p1_x = p1[0]
    globals.safe_zone_p1_y = p1[1]
    globals.safe_zone_p2_x = p2[0]
    globals.safe_zone_p2_y = p2[1]
    globals.safe_zone_p3_x = p3[0]
    globals.safe_zone_p3_y = p3[1]

    pts=np.array([p0, p1, p3, p2],np.int32)
    pts=pts.reshape((-1,1,2))

    cv2.fillPoly(safe_zone_mask,[pts],(188))

    _, counters, hierarchy = cv2.findContours(safe_zone_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #counters, hierarchy = cv2.findContours(safe_zone_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for index, tmp in enumerate(carsPos):
        if tmp is not np.nan:
            # init state
            state = 0

            # check car whether exist in safe zone
            p0 = tmp[0]
            p1 = tmp[1]
            p2 = tmp[2]
            p3 = tmp[3]
            cv2.circle(safe_zone_mask, p0, 4, (255), -1)
            cv2.circle(safe_zone_mask, p1, 4, (255), -1)
            cv2.circle(safe_zone_mask, p2, 4, (255), -1)
            cv2.circle(safe_zone_mask, p3, 4, (255), -1)
            res0 = cv2.pointPolygonTest(counters[0], p0, False)
            res1 = cv2.pointPolygonTest(counters[0], p1, False)
            res2 = cv2.pointPolygonTest(counters[0], p2, False)
            res3 = cv2.pointPolygonTest(counters[0], p3, False)

            # PIP for small picture
            p0_starty = p0[0] if p0[0]<p3[0] else p3[0]
            p0_startx = p0[1] if p0[1]<p1[1] else p1[1]

            p2_endy = p2[0] if p1[0]<p2[0] else p1[0]
            p2_endx = p2[1] if p3[1]<p2[1] else p3[1]

            if res0!=-1 or res1!=-1 or res2!=-1 or res3!=-1 :
                print(res0, res1, res2, res3)
                state = 1
                safe_zone_color = [255, 153, 18]
                
                # save short_distance_car for training data 
                globals.short_distance_car_p0_x = p0[0]
                globals.short_distance_car_p0_y = p0[1]
                globals.short_distance_car_p1_x = p1[0]
                globals.short_distance_car_p1_y = p1[1]
                globals.short_distance_car_p2_x = p2[0]
                globals.short_distance_car_p2_y = p2[1]
                globals.short_distance_car_p3_x = p3[0]
                globals.short_distance_car_p3_y = p3[1]

                # FIXME : NO TIME TO DEVELOPMENT, SO WE GIVE ASSUMPTION THAT ONLY ONE CAR IS DANGER IN SAFE ZONE
                if danger_ticktime >= frame_rate*2 and int(kmh)>40:
                    find_danger_driver = 1
                    safe_zone_color = [255, 0, 0]
                    state = 2

                danger_ticktime = danger_ticktime + 1
                
                # save label for training data
                globals.label = state

                # Assign license id
                danger_candidate_id = lp_strs[index]
                print("assign danger_candidate_id:", danger_candidate_id)
            else:
                # It means NO CAR exist in safe zone
                #if danger_candidate_id == lp_strs[index]:
                # FIXME : sometimes, car license ID cannot be detected, so we need to workaround this case
                
                # save short_distance_car for training data 
                globals.short_distance_car_p0_x = -1
                globals.short_distance_car_p0_y = -1
                globals.short_distance_car_p1_x = -1
                globals.short_distance_car_p1_y = -1
                globals.short_distance_car_p2_x = -1
                globals.short_distance_car_p2_y = -1
                globals.short_distance_car_p3_x = -1
                globals.short_distance_car_p3_y = -1
                
                # save label for training data
                globals.label = 0
                
                if find_danger_driver == 1:
                    find_danger_driver = 0 
                    globals.save_danger_driver_picture = 1
                    state = 0
                    danger_ticktime = 0

            # PACK all information for PIP
            car_detected_list.append([state, (p0_startx, p0_starty), (p2_endx, p2_endy), lp_strs[index]])


    #cv2.imwrite('safe_zone_mask.jpg', safe_zone_mask)

    cv2.polylines(overlay,[pts],True,(0,0,255),5)
    cv2.fillPoly(overlay,[pts],safe_zone_color)

    cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0, output)

    return output, car_detected_list

def removeCar(img, carPos):
    for tmp in carPos :
        startX = tmp[0][0]
        startY = tmp[0][1]

        if startX < 0:
            startX = 0
        if startY < 0:
            startY = 0

        lenX = tmp[1][0]-startX
        lenY = tmp[3][1]-startY
        img[startY:startY+lenY, startX:startX+lenX] = 0
    return img

def draw_cars(img, carsPos, lpsPos, lp_strs, Tcars, Ccars):

    # cars
    for index, tmp in enumerate(carsPos):
        p0 = tmp[0]
        p1 = tmp[1]
        p2 = tmp[2]
        p3 = tmp[3]

        pts=np.array([p0, p1, p2, p3],np.int32)
        pts=pts.reshape((-1,1,2))
        fontColor = (255, 255, 0)
        fontSize=1

        # car types
        if Ccars != [] and Ccars[index] == 'luxury':
            font = cv2.FONT_HERSHEY_SIMPLEX
            fontColor=1
            fontColor = (255, 0, 0)
            p0 = (p0[0], p0[1]-7)
            cv2.putText(img, Ccars[index], p0, font, fontSize, fontColor, 2)
        else :
            font = cv2.FONT_HERSHEY_SIMPLEX
            p0 = (p0[0], p0[1]-7)
            cv2.putText(img, str(Tcars[index].decode('utf-8')), p0, font, fontSize, fontColor, 2)
        
        cv2.polylines(img,[pts],True,fontColor,2)

    # plate
    for index, tmp in  enumerate(lpsPos):

        if tmp is not np.nan:
            p0 = tmp[0]
            p1 = tmp[1]
            p2 = tmp[2]
            p3 = tmp[3]

            pts=np.array([p0, p1, p2, p3],np.int32)
            pts=pts.reshape((-1,1,2))
            cv2.polylines(img,[pts],True,(0,0,250),5)

            if lp_strs[index] is not np.nan:
                font = cv2.FONT_HERSHEY_SIMPLEX
                fontColor = (255, 0, 255)
                fontSize=1
                p0 = (p0[0], p0[1]-7)
                cv2.putText(img, lp_strs[index], p0, font, fontSize, fontColor, 2)

    return img



def lane_pipeline(img, carsPos, lpsPos, lp_strs, Tcars, Ccars, kmh):

    perspective_ratio = np.float32([(0.41,0.78),(0.47,0.78),(0.2,1),(0.8,1)]),
    global running_avg
    global index

    width = img.shape[0]
    length = img.shape[1]

    img_ = pipeline(img)

    # remove car part for lane detection
    img_ = removeCar(img_, carsPos)

    # debug
    #cv2.imwrite('debug1.jpg', 255*img_)

    img_, debug_img = perspective_warp(img, img_, dst_size=(length,width), src=perspective_ratio)

    # debug
    #cv2.imwrite('debug2.jpg', 255*img_)
    #cv2.imwrite('debug3.jpg', debug_img )

    out_img, curves, lanes, ploty = sliding_window(img_, draw_windows=False)

    # debug
    #cv2.imwrite('debug4.jpg', out_img)

    # draw lanes
    curverad =get_curve(img, curves[0], curves[1])
    lane_curve = np.mean([curverad[0], curverad[1]])
    img = draw_lanes(img, curves[0], curves[1], perspective_ratio=perspective_ratio)

    # draw cars & car  plate
    img = draw_cars(img, carsPos, lpsPos, lp_strs, Tcars, Ccars)

    # debug
    #cv2.imwrite('debug5.jpg', img)

    img, car_detected_list = draw_safe_zone(img, carsPos, lp_strs, kmh)
    #cv2.imwrite('safe_zone.jpg', img)

    return img, car_detected_list

