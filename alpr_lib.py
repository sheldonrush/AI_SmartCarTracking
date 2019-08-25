
# coding: utf-8

# In[1]:

import math
import cv2
import numpy as np
import re
import sys
import csv
import functools
import keras

import darknet.python.darknet as dn

from numpy.linalg import norm
from src.label import Label, lwrite
from os.path import splitext, basename, isdir, isfile
from os import makedirs
from src.utils import crop_region, image_files_from_folder
from src.utils import im2single
from src.label import Shape, writeShapes
from src.label import dknet_label_conversion
from src.utils import nms
from src.utils import crop_region, image_files_from_folder
from src.drawing_utils import draw_label, draw_text, draw_text2, draw_losangle, write2img
from src.label import lread, Label, readShapes
from darknet.python.darknet import detect2
from src.keras_utils import load_model, detect_lp

from keras.preprocessing import image as keras_image
from tensorflow.keras.models import load_model as keras_load_model
from keras.applications.inception_resnet_v2 import preprocess_input as resnet_preprocess_input

# In[2]:

vehicle_threshold = .6
vehicle_weights = 'data/vehicle-detector/yolov3.weights'.encode("utf-8")
vehicle_netcfg  = 'data/vehicle-detector/yolov3.cfg'.encode("utf-8")
vehicle_dataset = 'data/vehicle-detector/coco.data'.encode("utf-8")    
vehicle_net  = dn.load_net(vehicle_netcfg, vehicle_weights, 0)
vehicle_meta = dn.load_meta(vehicle_dataset) 

vehicle_width_threshold = 60 # pixels
vehicle_height_threshold = 60 # pixels

# vehicle_weights = 'data/vehicle-detector/yolo-voc.weights'.encode("utf-8")
# vehicle_netcfg  = 'data/vehicle-detector/yolo-voc.cfg'.encode("utf-8")
# vehicle_dataset = 'data/vehicle-detector/voc.data'.encode("utf-8")    

lp_threshold = .6
wpod_net_path = "data/lp-detector/wpod-net_update1.h5"
wpod_net = load_model(wpod_net_path)

ocr_threshold = .6
ocr_weights = 'data/ocr/ocr-net.weights'.encode("utf-8")
ocr_netcfg  = 'data/ocr/ocr-net.cfg'.encode("utf-8")
ocr_dataset = 'data/ocr/ocr-net.data'.encode("utf-8")
ocr_net  = dn.load_net(ocr_netcfg, ocr_weights, 0)
ocr_meta = dn.load_meta(ocr_dataset)

car_cat_threshold = .75
car_cat_net_path = "data/car-cat-detector/car-cat-detector.h5"
car_cat_net = keras_load_model(car_cat_net_path)

car_cat_img_size = 299
car_cat_vehicle_width_threshold = 120 # pixels
car_cat_vehicle_height_threshold = 120 # pixels

car_cat_names = []
with open('data/car-cat-detector/car-cat-detector.names') as csvDataFile:
    csvReader = csv.reader(csvDataFile, delimiter=';')
    for row in csvReader:
        car_cat_names.append(row[0])

print(car_cat_names)

CAR_TRACK_MIN_DISTANCE = 40
CAR_TRACK_N_FRAME = 8
CAR_TRACK_N_LPS = 4
CAR_TRACK_N_CAR_CAT = 15 

#global
ref_n_frame_axies = []
ref_n_frame_label = []
ref_n_frame_axies_flatten = []
ref_n_frame_label_flatten = []
label_cnt = 1
lp_strs_dict = {}
car_cat_dict = {}
#~

def vehicle_detect(vehicle_net, vehicle_meta, vehicle_threshold, img):
    print ('Searching for vehicles using YOLO...')

    R,_ = detect2(vehicle_net, vehicle_meta, img, thresh=vehicle_threshold)
    R = [r for r in R if r[0] in [b'car',b'bus']]
    print ('\t\t%d cars found' % len(R))
    Icars = []
    Lcars = []
    Tcars = []
    Scars = []
    Pcars = []
    if len(R):
        Iorig = img
        WH = np.array(Iorig.shape[1::-1],dtype=float)

        for i,r in enumerate(R):
#             print ('\t\t\t%d %d %4.2f' % (r[2][2], r[2][3], r[1]))
            cx,cy,w,h = (np.array(r[2])/np.concatenate( (WH,WH) )).tolist()
            tl = np.array([cx - w/2., cy - h/2.])
            br = np.array([cx + w/2., cy + h/2.])
            label = Label(0,tl,br)
            Icar = crop_region(Iorig,label)
            if (r[2][2] > vehicle_width_threshold and r[2][3] > vehicle_height_threshold):
                Icars.append(Icar)
                Lcars.append(label)
                Tcars.append(r[0])
                Scars.append(r[1])
                Pcars.append(r[2])
            
            #cv2.imwrite('%s/%dcar.png' % (car_img_dir,i),Icar)
        #lwrite('%s/cars.txt' % (car_img_dir),Lcars)

    print ('\t\t%d cars chosen' % len(Icars))
    return Icars, Lcars, Tcars, Scars, Pcars


def car_cat_detect(car_cat_net, car_cat_threshold, Icars, Tcars):
    print ('Searching for car cat using CAR-CAT-NET')
    Ccars = []
    lux_car_found = 0
    for i, (Icar, Tcar) in enumerate(zip(Icars, Tcars)):
        W, H = Icar.shape[:2]
        if (str(Tcar.decode("utf-8"))=='car' and W > car_cat_vehicle_width_threshold and H > car_cat_vehicle_width_threshold):
            print ('\t Processing car %s' % i)
            img = cv2.resize(Icar,(car_cat_img_size,car_cat_img_size),interpolation=cv2.INTER_CUBIC)
            img = keras_image.img_to_array(img)
            img = resnet_preprocess_input(img)  
            img = np.array(img, dtype=np.float32) 
            img = np.expand_dims(img, axis=0)
#             print (img.shape)
            preds = car_cat_net.predict(img)
#             print (preds[0])
            the_pred = np.argmax(preds[0])
            pred_name = car_cat_names[the_pred]
            val_pred = max(preds[0])
            print ('\t\t car cat: %s, prob: %4.2f' % (pred_name, val_pred))
            if ((val_pred > car_cat_threshold) and not(pred_name=='sedan' or pred_name=='suv' or pred_name=='hatchback')):
                car_cat = 'luxury'
            else:
                car_cat = 'regular'
            
#             print ('\t\t car cat: %s' % (car_cat))
            Ccars.append(car_cat)         
        else:
            Ccars.append(np.nan)
    
    return Ccars
# In[4]:


def plate_detect(wpod_net, lp_threshold, Icars, Lcars):
    print ('Searching for license plates using WPOD-NET')
    Ilps = []
    Llps = []
    for i, (Icar, Lcar) in enumerate(zip(Icars, Lcars)):
        print ('\t Processing car %s' % i)
        ratio = float(max(Icar.shape[:2]))/min(Icar.shape[:2])
        side  = int(ratio*288.)
        bound_dim = min(side + (side%(2**4)),608)
#         print ("\t\tBound dim: %d, ratio: %f" % (bound_dim,ratio))
        Icar = Icar.astype('uint8')
        Llp,LlpImgs,_ = detect_lp(wpod_net,im2single(Icar),bound_dim,2**4,(240,80),lp_threshold)
        if len(LlpImgs):
            print ("\t\tLP found")
            Ilp = LlpImgs[0]
            Ilp = cv2.cvtColor(Ilp, cv2.COLOR_BGR2GRAY)
            Ilp = cv2.cvtColor(Ilp, cv2.COLOR_GRAY2BGR)
            s = Shape(Llp[0].pts)
            #cv2.imwrite('%s/car%s_lp.png' % (plate_img_dir, i),Ilp*255.)
            #writeShapes('%s/car%s_lp.txt' % (plate_img_dir, i),[s]) 
            Ilps.append(Ilp*255.)
            Llps.append(s)
        else:
            Ilps.append(np.nan)
            Llps.append(np.nan)            
    return Ilps, Llps


# In[5]:


def ocr_detect(ocr_net, ocr_meta, Ilps):
    print ('Performing OCR...')
    lp_strs = []
    for i, Ilp in enumerate(Ilps):
        if (Ilp is not np.nan):
            print ('\tScanning car %s' % i)
            R,(width,height) = detect2(ocr_net, ocr_meta, Ilp, thresh=ocr_threshold, nms=None)
            if len(R):
                L = dknet_label_conversion(R,width,height)
                L = nms(L,.45)
                L.sort(key=lambda x: x.tl()[0])
                lp_str = ''.join([chr(l.cl()) for l in L])
                sys.stdout.write('\t\t%s\n' % lp_str)
                regex = re.compile(r'([0-9|I]{4}[A-Z][A-Z|0-9])|([A-Z]{3}[0-9|I]{4})')
                match = regex.search(lp_str)
                sys.stdout.write('\t\t%s\n' % match)
                if ((len(lp_str)==6 or len(lp_str)==7) and match is not None):
                    lp_strs.append(lp_str)
                else:
                    lp_strs.append(np.nan)         
            else:
                lp_strs.append(np.nan)
        else:
            lp_strs.append(np.nan)
    return lp_strs


# In[6]:


def draw_car(img, Lcars, Tcars, Ccars, IDcars, Llps, lp_strs):
    print ('Performing outputting car ...')
    
    YELLOW = (  0,255,255)
    RED    = (  0,  0,255)  
    WHITE  = (255,255,255)

    I = img

    if Lcars:
        for i,(lcar, tcar, ccar, idcar) in enumerate(zip(Lcars, Tcars, Ccars, IDcars)):
            draw_label(I,lcar,color=YELLOW,thickness=3)
            if (ccar is not np.nan):
                car_cat = str(tcar.decode("utf-8")) + " " + ccar + " " + str(idcar)
            else:
                car_cat = str(tcar.decode("utf-8")) + " " + str(idcar)
            draw_text(I,lcar,car_cat,color=YELLOW,thickness=1)
   
            lp_label = Llps[i]
            lp_label_str = lp_strs[i]

            if (lp_label is not np.nan):
                pts = lp_label.pts*lcar.wh().reshape(2,1) + lcar.tl().reshape(2,1)
                ptspx = pts*np.array(I.shape[1::-1],dtype=float).reshape(2,1)
                draw_losangle(I,ptspx,RED,3)

                if (lp_label_str is not np.nan):
                    lp_str = lp_label_str.strip()
                    llp = Label(0,tl=pts.min(1),br=pts.max(1))
                    write2img(I,llp,lp_str)
                    sys.stdout.write('%s\n' % lp_str)
            else:
                if (lp_label_str is not np.nan):
                    lp_str = lp_label_str.strip()
                    draw_text2(I,lcar,lp_str,color=WHITE,thickness=3)
                    sys.stdout.write('%s\n' % lp_str)

    return I

def carLabel2pos(I, Lcars, Llps):
    carsPos = []
    lpsPos = []
    for i, Lcar in enumerate(Lcars):
        #car position
        wh = np.array(I.shape[1::-1]).astype(float)
        tl = tuple((Lcar.tl()*wh).astype(int).tolist())
        br = tuple((Lcar.br()*wh).astype(int).tolist())
        tr = (br[0], tl[1])
        bl = (tl[0], br[1])
        carsPos.append((tl, tr, br, bl))
        #lp position
        if (Llps[i] is not np.nan):
            pts = Llps[i].pts*Lcar.wh().reshape(2,1) + Lcar.tl().reshape(2,1)
            ptspx = pts*np.array(I.shape[1::-1],dtype=float).reshape(2,1)
            assert(ptspx.shape[0] == 2 and ptspx.shape[1] == 4)
            lpPos = []
            for i in range(4):
                pt = tuple(ptspx[:,i].astype(int).tolist())
                lpPos.append(pt)
            lpsPos.append(lpPos)
        else:
            lpsPos.append(np.nan)
        
    return carsPos, lpsPos


def car_track(Pcars, Ccars, lp_strs, min_distance=CAR_TRACK_MIN_DISTANCE):
    global ref_n_frame_axies
    global ref_n_frame_label
    global ref_n_frame_axies_flatten
    global ref_n_frame_label_flatten    
    global label_cnt 
    global lp_strs_dict
    global car_cat_dict
    cur_frame_axies = []
    cur_frame_label = []
    cur_frame_lp_strs = []
    cur_frame_car_cat = []
    for (Pcar, Ccar, lp_str) in zip(Pcars, Ccars, lp_strs):
        x = Pcar[0]
        y = Pcar[1]
        cur_lp_str = np.nan
        cur_car_cat = np.nan
        lbl = float('nan')
        if(len(ref_n_frame_label_flatten) > 0):
            b = np.array([(x,y)])
            a = np.array(ref_n_frame_axies_flatten)
            distance = norm(a-b,axis=1)
            min_value = distance.min()
            if(min_value < min_distance):
                idx = np.where(distance==min_value)[0][0]
                lbl = ref_n_frame_label_flatten[idx]
        if(math.isnan(lbl)):
            lbl = label_cnt
            label_cnt += 1
            
#         print("lbl ", str(lbl), ",lp_str ", str(lp_str), ",Ccar ", str(Ccar))
        
        if (str(lbl) in lp_strs_dict):
            if (lp_strs_dict[str(lbl)][1] < CAR_TRACK_N_LPS):
                if (lp_strs_dict[str(lbl)][0] == lp_str):
                    lp_strs_dict[str(lbl)][1] += 1
                elif (lp_str is not np.nan):
                    lp_strs_dict[str(lbl)] = [lp_str, 1]                    
        else:
            if (lp_str is not np.nan):
                lp_strs_dict[str(lbl)] = [lp_str, 1]    
                
        if (str(lbl) in car_cat_dict):
            if (car_cat_dict[str(lbl)][1] < CAR_TRACK_N_CAR_CAT):
                if (car_cat_dict[str(lbl)][0] == Ccar):
                    car_cat_dict[str(lbl)][1] += 1
                elif (Ccar is not np.nan):
                    car_cat_dict[str(lbl)] = [Ccar, 1]                    
        else:
            if (Ccar is not np.nan):
                car_cat_dict[str(lbl)] = [Ccar, 1]   
                
        if (str(lbl) in lp_strs_dict):
            if (lp_strs_dict[str(lbl)][1]==CAR_TRACK_N_LPS):
                cur_lp_str = lp_strs_dict[str(lbl)][0]
                
        if (str(lbl) in car_cat_dict):
            if (car_cat_dict[str(lbl)][1]==CAR_TRACK_N_CAR_CAT):
                cur_car_cat = car_cat_dict[str(lbl)][0]
            
#         print("lp_strs_dict ", lp_strs_dict)
#         print("car_cat_dict ", car_cat_dict)
        cur_frame_label.append(lbl)
        cur_frame_axies.append((x,y))
        cur_frame_lp_strs.append(cur_lp_str)
        cur_frame_car_cat.append(cur_car_cat)
    if(len(ref_n_frame_axies) == CAR_TRACK_N_FRAME):
        del ref_n_frame_axies[0]
        del ref_n_frame_label[0]
    ref_n_frame_label.append(cur_frame_label)
    ref_n_frame_axies.append(cur_frame_axies)    
    ref_n_frame_axies_flatten = [a for ref_n_frame_axie in ref_n_frame_axies for a in ref_n_frame_axie]
    ref_n_frame_label_flatten = [b for ref_n_frame_lbl in ref_n_frame_label for b in ref_n_frame_lbl]
    
#     print("cur_frame_label ", cur_frame_label)
#     print("ref_n_frame_label ", ref_n_frame_label)
#     print("ref_n_frame_label_flatten ", ref_n_frame_label_flatten)
    
#     print("cur_frame_axies ", cur_frame_axies)
#     print("ref_n_frame_axies ", ref_n_frame_axies)
#     print("ref_n_frame_axies_flatten ", ref_n_frame_axies_flatten)
    
    print("cur_frame_lp_strs ", cur_frame_lp_strs)
    print("cur_frame_car_cat ", cur_frame_car_cat)
    
    return cur_frame_label, cur_frame_lp_strs, cur_frame_car_cat
                    

def alpr(img):
    Icars, Lcars, Tcars, Scars, Pcars = vehicle_detect(vehicle_net, vehicle_meta, vehicle_threshold, img)
    Ccars = car_cat_detect(car_cat_net, car_cat_threshold, Icars, Tcars)
    Ilps, Llps = plate_detect(wpod_net, lp_threshold, Icars, Lcars)
    lp_strs = ocr_detect(ocr_net, ocr_meta, Ilps)
    IDcars, lp_strs, Ccars = car_track(Pcars, Ccars, lp_strs)
    print ("IDcars ", IDcars)
    print ("lp_strs ", lp_strs)
    print ("Ccars ", Ccars)
    Idraw = draw_car(img, Lcars, Tcars, Ccars, IDcars, Llps, lp_strs)
    carsPos, lpsPos = carLabel2pos(img, Lcars, Llps)
    # Remove the car with long distance
    temp = carsPos
    for index, pos in enumerate(temp):
        polygon = np.array([[pos[0], pos[1], pos[2], pos[3]]], dtype=np.int32)
        im = np.zeros(img.shape[:2], dtype="uint8")
        polygon_mask = cv2.fillPoly(im, polygon, 255)
        area = np.sum(np.greater(polygon_mask, 0))
        if area < 750:
            del carsPos[index]
            del lpsPos[index]
            del lp_strs[index]
            del Tcars[index]
            print("pos:{0} , area:{1}".format(pos, area))

    return Idraw, carsPos, lpsPos, lp_strs, Tcars, Ccars, IDcars

