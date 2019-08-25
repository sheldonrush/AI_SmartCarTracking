import cv2
import numpy as np
import globals

def make_dashboard(ori_img, out_img, kmh, NNgrid1, NNgrid2, NNgrid3, EEgrid1, EEgrid2, EEgrid3, car_detected_list):

    # save danger driver picture in persist list
    if globals.save_danger_driver_picture == 1:
        globals.save_danger_driver_picture = 0
        globals.catch_danger_driver.append(globals.tmp_roi) 

    ## Dashboard Info 
    # Latitude and longitude 
    p0 = (30, 30)
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontColor = (0, 0, 255)
    fontSize=0.8
    str1 = NNgrid1 +'N'+ NNgrid2+'\''+ NNgrid3+'\"'
    str2 = EEgrid1 +'E'+ EEgrid2+'\''+ EEgrid3+'\"'
    cv2.putText(out_img, '{0},{1}'.format(str1, str2), p0, font, fontSize, fontColor, 2)
    
    # Self-Speed
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontColor = (0, 0, 255)
    fontSize=1
    p0_speed = (p0[0], p0[1]+50)
    cv2.putText(out_img, 'speed: {0} km/hr'.format(kmh), p0_speed, font, fontSize, fontColor, 2)
    # save self-speed for training data 
    globals.self_speed = int(kmh)
    
    # Danger driver cnt
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontColor = (255, 0, 0)
    fontSize=1
    p0_danger_driver_cnt = (p0_speed[0], p0_speed[1]+50)
    cv2.putText(out_img, 'Bastard: {0}'.format(len(globals.catch_danger_driver)), p0_danger_driver_cnt, font, fontSize, fontColor, 2)
    
    # detected car cnt
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontColor = (0, 0, 255)
    fontSize=1
    p0_car_cnt = (p0_danger_driver_cnt[0], p0_danger_driver_cnt[1]+50)
    cv2.putText(out_img, 'Car: {0}'.format(len(car_detected_list)), p0_car_cnt, font, fontSize, fontColor, 2)
    # save detected car cnt for training data 
    globals.detected_car_num = len(car_detected_list)

    ## Info Line
    p0_line = (p0[0]+300, 0)
    p1_line = (p0[0]+300, p0[1]+210)
    Info_line_color = (0xb2, 0x22, 0x22)
    cv2.line(out_img, p0_line, p1_line, Info_line_color, 10)

    ## draw cars
    width = 220
    height = 220
    dim = (width-8, height-8) # for boarder pixels

    draw_car_p = [p0_line[0]+15, 3]

    # draw persist danger driver picture first
    for temp in globals.catch_danger_driver:
        
        if temp == []:
            continue

        (roi, license_number) = temp
        
        if draw_car_p[0]+width > ori_img.shape[1]:
            # out of x-resolution
            break
        if roi == []:
            break

        out_img[draw_car_p[1]:draw_car_p[1]+height, draw_car_p[0]: draw_car_p[0]+width] = roi
        
        # draw license number
        if license_number is not np.nan:
            font = cv2.FONT_HERSHEY_SIMPLEX
            fontColor = (255, 255, 255)
            fontSize=1
            cv2.putText(out_img, license_number, (draw_car_p[0], draw_car_p[1]+height+1), font, fontSize, fontColor, 2)
        
        draw_car_p = [draw_car_p[0]+width+5, draw_car_p[1]]
    
    # draw dynamical detected car
    for i in range(len(car_detected_list)):
        state = car_detected_list[i][0]
        p0 = car_detected_list[i][1]
        p1 = car_detected_list[i][2]
        license_number = car_detected_list[i][3]

        # Handle Boundary case
        if p0[1]<0:
            print(p0)
            p0 = (p0[0],0)
        if p1[1]<0:
            p1 = (p1[0],0)
            print(p1)

        car_roi = ori_img[p0[0]:p1[0], p0[1]:p1[1]]

        # resize image
        if state == 0:
            color = [0, 255, 0]
        elif state == 1:
            color = [255, 153, 18]
        else:
            color = [255, 0, 0]

        car_roi = cv2.resize(car_roi, dim, interpolation = cv2.INTER_AREA)
        car_roi = cv2.copyMakeBorder(car_roi,4,4,4,4,cv2.BORDER_CONSTANT,value=color)

        # keep the small picture of danger driver
        if state == 2:
            globals.tmp_roi = (car_roi,license_number)
        else:
            pass
        

        # draw car
        if out_img.shape[0]<draw_car_p[1]+height or out_img.shape[1]<draw_car_p[0]+width:
            continue
        else:
            out_img[draw_car_p[1]:draw_car_p[1]+height, draw_car_p[0]: draw_car_p[0]+width] = car_roi

        # draw license number
        if license_number is not np.nan:
            font = cv2.FONT_HERSHEY_SIMPLEX
            fontColor = (255, 255, 255)
            fontSize=1
            cv2.putText(out_img, license_number, (draw_car_p[0], draw_car_p[1]+height+1), font, fontSize, fontColor, 2)

        draw_car_p = [draw_car_p[0]+width+5, draw_car_p[1]]

    #cv2.imwrite('dashboard.jpg', out_img)

    return out_img
