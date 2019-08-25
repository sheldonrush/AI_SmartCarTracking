from lane_detection import lane_pipeline
from alpr_lib import alpr
from moviepy.editor import VideoFileClip
from car_dashboard import make_dashboard
from read_gps import readGPS
import sys
import globals
import pandas as pd 
import os
import glob
import numpy as np
import math 

## Parameter Setting
# file_name = 'FILE190810-110632F'
# file_name = 'FILE190810-100617F'
# start_time = 50 # unit :second
# end_time = 100 # unit :second 177->45


file_name = 'FILE190810-103932F'
start_time = 0 # unit :second
end_time = 140 # unit :second 177->45

# file_name = 'FILE190810-104832F'
# file_name = 'FILE190810-105132F'
# file_name = 'FILE190810-103031F'
# file_name = 'FILE190810-103332F'
# file_name = 'FILE190810-103631F'
# file_name = 'FILE190810-104532F'
# file_name = 'FILE190810-104832F'
# file_name = 'FILE190810-110332F'
# file_name = 'FILE190811-124346F'
# file_name = 'FILE190810-110632F'
# file_name = 'FILE190810-112432F'

input_video_file = 'test_video/{0}.MP4'.format(file_name)
input_gps_file = 'test_video/{0}.NMEA'.format(file_name)
video_output = 'result/video/{0}.MP4'.format(file_name)

# Global Variable
frame_rate = 30

# Gathering data for training 
self_speed_list = []
detected_car_nums_list = []
short_distance_car_p0_x_list = []
short_distance_car_p0_y_list = []
short_distance_car_p1_x_list = []
short_distance_car_p1_y_list = []
short_distance_car_p2_x_list = []
short_distance_car_p2_y_list = []
short_distance_car_p3_x_list = []
short_distance_car_p3_y_list = []
safe_zone_p0_x_list = []
safe_zone_p0_y_list = []
safe_zone_p1_x_list = []
safe_zone_p1_y_list = []
safe_zone_p2_x_list = []
safe_zone_p2_y_list = []
safe_zone_p3_x_list = []
safe_zone_p3_y_list = []
label_list = []
fcnt_list = []

# Debug
# clip partial video
# start_time = 65 # unit :second
# end_time = 120 # unit :second 177->45

# Image Processing
def vid_pipeline(img):
    global start_time

    # get speed
    print('start_time : {0}, kmhs length : {1}'.format(start_time, len(globals.kmhs)))
    
    # Boundary case
    if (start_time+math.floor(globals.frm_cnt/frame_rate)) == len(globals.kmhs):
        return img
    
    kmh = globals.kmhs[start_time+math.floor(globals.frm_cnt/frame_rate)]
    NNgrid1 = globals.Nlist1[start_time+math.floor(globals.frm_cnt/frame_rate)]
    NNgrid2 = globals.Nlist2[start_time+math.floor(globals.frm_cnt/frame_rate)]
    NNgrid3 = globals.Nlist3[start_time+math.floor(globals.frm_cnt/frame_rate)]
    EEgrid1 = globals.Elist1[start_time+math.floor(globals.frm_cnt/frame_rate)]
    EEgrid2 = globals.Elist2[start_time+math.floor(globals.frm_cnt/frame_rate)]
    EEgrid3 = globals.Elist3[start_time+math.floor(globals.frm_cnt/frame_rate)]

    ori_img = img.copy()

    # car detection & plate detection
    Idraw, carsPos, lpsPos, lp_strs, Tcars, Ccars, Car_Id = alpr(img)

    # lane detection
    out_img, car_detected_list = lane_pipeline(ori_img, carsPos, lpsPos, lp_strs, Tcars, Ccars, kmh)

    # dashboard
    out_img = make_dashboard(ori_img, out_img, kmh, NNgrid1, NNgrid2, NNgrid3, EEgrid1, EEgrid2, EEgrid3, car_detected_list)
    #out_img = make_dashboard(ori_img, ori_img, kmh, car_detected_list)

    # Gathering data for training 
    self_speed_list.append(globals.self_speed)
    detected_car_nums_list.append(globals.detected_car_num)
    short_distance_car_p0_x_list.append(globals.short_distance_car_p0_x)
    short_distance_car_p0_y_list.append(globals.short_distance_car_p0_y)
    short_distance_car_p1_x_list.append(globals.short_distance_car_p1_x)
    short_distance_car_p1_y_list.append(globals.short_distance_car_p1_y)
    short_distance_car_p2_x_list.append(globals.short_distance_car_p2_x)
    short_distance_car_p2_y_list.append(globals.short_distance_car_p2_y)
    short_distance_car_p3_x_list.append(globals.short_distance_car_p3_x)
    short_distance_car_p3_y_list.append(globals.short_distance_car_p3_y)
    safe_zone_p0_x_list.append(globals.safe_zone_p0_x)
    safe_zone_p0_y_list.append(globals.safe_zone_p0_y)
    safe_zone_p1_x_list.append(globals.safe_zone_p1_x)
    safe_zone_p1_y_list.append(globals.safe_zone_p1_y)
    safe_zone_p2_x_list.append(globals.safe_zone_p2_x)
    safe_zone_p2_y_list.append(globals.safe_zone_p2_y)
    safe_zone_p3_x_list.append(globals.safe_zone_p3_x)
    safe_zone_p3_y_list.append(globals.safe_zone_p3_y)
    label_list.append(globals.label)
    fcnt_list.append(globals.frm_cnt)
    
    print('frame cnt :', globals.frm_cnt)
    globals.frm_cnt = globals.frm_cnt +1

    return out_img

if __name__ == '__main__':
    
    argc = len(sys.argv)
    if argc == 1:
        option = 'demo'
    else :
        option = sys.argv[1]
    
    if option == 'dataset':
        for index, file in enumerate(glob.glob("test_video/*.MP4")):
            print(index, file)
            name = file[11:29]
            input_video_file = 'test_video/{0}.MP4'.format(name)
            input_gps_file = 'test_video/{0}.NMEA'.format(name)
            video_output = 'result/video/{0}.MP4'.format(name)
            dataset_output = 'result/data_set/{0}.csv'.format(name)
            print(input_video_file)
    
            # Initialize globals
            globals.Initialize() 

            # Read GPS file for self-speeding
            readGPS(input_gps_file)

            # clip partial video for developing
#             myclip = VideoFileClip(input_video_file).subclip(start_time,end_time)
            myclip = VideoFileClip(input_video_file)
            # GOOGOGOGO
            clip = myclip.fl_image(vid_pipeline)

            # Output the file for lane detection
            clip.write_videofile(video_output , audio=False)
    
            # Gathering data for training 
            DataSet = list(zip(fcnt_list, self_speed_list, detected_car_nums_list, short_distance_car_p0_x_list, short_distance_car_p0_y_list, short_distance_car_p1_x_list, short_distance_car_p1_y_list, short_distance_car_p2_x_list, short_distance_car_p2_y_list, short_distance_car_p3_x_list, short_distance_car_p3_y_list, safe_zone_p0_x_list, safe_zone_p0_y_list, safe_zone_p1_x_list, safe_zone_p1_y_list, safe_zone_p2_x_list, safe_zone_p2_y_list, safe_zone_p3_x_list, safe_zone_p3_y_list, label_list))
            df = pd.DataFrame(DataSet, columns=['frame_cnt', 'self_speed', 'detected_car_nums', 'short_distance_car_p0_x','short_distance_car_p0_y','short_distance_car_p1_x','short_distance_car_p1_y','short_distance_car_p2_x','short_distance_car_p2_y','short_distance_car_p3_x','short_distance_car_p3_y','safe_zone_p0_x', 'safe_zone_p0_y','safe_zone_p1_x', 'safe_zone_p1_y','safe_zone_p2_x', 'safe_zone_p2_y','safe_zone_p3_x', 'safe_zone_p3_y', 'label'])
        
            print('frm_cnt = ', globals.frm_cnt)
            df.to_csv(dataset_output, index = True)
    else:
    # Initialize globals
        globals.Initialize() 

        # Read GPS file for self-speeding
        readGPS(input_gps_file)

        # clip partial video for developing
        myclip = VideoFileClip(input_video_file).subclip(start_time,end_time)
#         myclip = VideoFileClip(input_video_file)
        # GOOGOGOGO
        clip = myclip.fl_image(vid_pipeline)

        # Output the file for lane detection
        clip.write_videofile(video_output , audio=False)
