def Initialize():
    global save_danger_driver_picture
    save_danger_driver_picture = 0
    global catch_danger_driver
    catch_danger_driver = []
    global tmp_roi
    tmp_roi = []
    global kmhs
    kmhs = []
    global Nlist1  #北緯度
    Nlist1 = []
    global Nlist2  #北緯分
    Nlist2 = []
    global Nlist3  #北緯秒
    Nlist3 = []
    global Elist1  #東經度
    Elist1 = []
    global Elist2  #東經分
    Elist2 = []
    global Elist3  #東經秒
    Elist3 = []
    global frm_cnt
    frm_cnt = 0
    
    
    # Pack information for training data
    global self_speed
    self_speed = 0 
    
    global detected_car_num
    detected_car_num = 0
    
    # p0
    global short_distance_car_p0_x
    short_distance_car_p0_x = -1
    global short_distance_car_p0_y
    short_distance_car_p0_y = -1
    
    # p1
    global short_distance_car_p1_x
    short_distance_car_p1_x = -1
    global short_distance_car_p1_y
    short_distance_car_p1_y = -1
    
    # p2
    global short_distance_car_p2_x
    short_distance_car_p2_x = -1
    global short_distance_car_p2_y
    short_distance_car_p2_y = -1
    
    # p3
    global short_distance_car_p3_x
    short_distance_car_p3_x = -1
    global short_distance_car_p3_y
    short_distance_car_p3_y = -1

    # safe zone 
    global safe_zone_p0_x
    safe_zone_p0_x = 0
    global safe_zone_p0_y
    safe_zone_p0_y = 0
    
    global safe_zone_p1_x
    safe_zone_p1_x = 0
    global safe_zone_p1_y
    safe_zone_p1_y = 0
    
    global safe_zone_p2_x
    safe_zone_p2_x = 0
    global safe_zone_p2_y
    safe_zone_p2_y = 0
    
    global safe_zone_p3_x
    safe_zone_p3_x = 0
    global safe_zone_p3_y
    safe_zone_p3_y = 0
    
    # labels 
    global label 
    label = 0
