"""
    headless script developped based on findings gained in detect_car.ipynb
    goal of this script is to find the largest YOLObox that has a label 'car', 'truck' or 'vehicle'
    and save the box coords in the SQL database.

    Under imports you can set parameters for batch processing. 

    Override for AMD gpus is required
    Override of lzma on linux is also required

"""
#basic modules
import os
from ultralytics import YOLO
from tqdm import tqdm
import math
##issue on UNIX systems with YOLO: https://github.com/ultralytics/yolov5/issues/1298

import sys
sys.path.append('utils')
from  configloader import Configloader
from database import Database
import car_detection as cd
from file_io import path_handler
import platform_dependency as plf

##Batch processing config: 
BATCH_SIZE = 250

detected_os = plf.get_platform()
if detected_os == 'Linux':
    print('A unix system was detected, please follow these steps to patch a known UNIX bug:')
    print('https://askubuntu.com/questions/1299255/how-can-i-solve-no-module-named-lzma')
    input('To continue press enter')

# Load the project configuration: 
config = Configloader()


detected_gpu = plf.get_gpu_info()
if detected_gpu == False:
    print(f'Notebook detected {detected_os} as OS and will use CPU for tasks where it can be done')
else:
    print(f'Notebook detected {detected_os} as OS and will use the available {detected_gpu} GPU')
    if detected_gpu != 'amd': 
        print(f"{detected_gpu} GPU's were not tested - code required workarounds to get working on AMD - no guarantees given.")

# Load the YOLO model
yolo_path = os.path.join(config.get('directories','root_dir'), config.get('directories', 'yolo_path'))
yolomodel = YOLO(yolo_path)  # YOLOv8 nano for speed, or 'yolov8s.pt' for more accuracy
yolomodel = YOLO(yolo_path)  # YOLOv8 nano for speed, or 'yolov8s.pt' for more accuracy
if detected_gpu == 'amd': 
    yolomodel = plf.yolo_override(yolomodel)

imdir = config.get('settings','image_directory')

#connect to database: 
connection_type = config.get('settings', 'connection')

user = config.get(connection_type, 'user')
pw = config.get(connection_type, 'pw')
host = config.get(connection_type, 'host')
db = config.get(connection_type, 'db')
port = int(config.get(connection_type, 'port'))
db = Database(host,
              port,
              user,
              pw,
              db
              )
db.connect()

#get basedir: 
basedir = config.get('settings', 'image_directory')
#slave functions for data type conversion (tested in experimental notebook and this is fine!)
def to_pixel(tensor): 
    v = math.ceil(tensor)
    return v 
def to_float(tensor):
    v = float(tensor)
    return v

#get images in batches of batchsize:
#find the lowest and highest id of non-processed-image rows:
lowquery = "SELECT min(id) AS minrange, max(id) AS maxrange FROM images WHERE processed = 0;"
imids = db.execute_query(lowquery)
for id in tqdm(range(imids[0]['minrange'], imids[0]['maxrange'], BATCH_SIZE)):
    batch = "SELECT id, image_path FROM images WHERE id BETWEEN %s AND %s AND processed = 0;"
    batch_args = [id, id+BATCH_SIZE]
    batch_images = db.execute_query(batch, batch_args)
    db.start_transaction()
    for image in batch_images:
        pk_id = image['id']
        path = path_handler(basedir, image['image_path'])
        if not os.path.exists(path):
            with open('logging/automated_yolo_tagger.txt', 'a+') as f:
                f.write(str(path)+ ' does not exists\n')
            continue
        is_car, image_path, box, confidence, area = cd.is_full_car(path, yolomodel)
        query = "UPDATE images SET "
        query_fields = []
        values = []

        if is_car:
            #get box coordinates
            x_min, y_min, x_max, y_max = box
            query_fields.append('use_image = 1')
            query_fields.append('yolobox_top_left_x = %s')
            query_fields.append('yolobox_top_left_y = %s')
            query_fields.append('yolobox_bottom_right_x = %s')
            query_fields.append('yolobox_bottom_right_y = %s')
            query_fields.append('area = %s')
            query_fields.append('confidence = %s')
            values.extend(
                [to_pixel(x_min), 
                 to_pixel(y_min), 
                 to_pixel(x_max), 
                 to_pixel(y_max), 
                 to_float(area), 
                 to_float(confidence)]
            )
        else:
            query_fields.append('use_image = 0')
        query_fields.append('processed = 1')
        
        values.append(pk_id)
        query = "UPDATE images SET " + ', '.join(query_fields) + " WHERE id = %s;"
        db.execute_query(query, values)
    db.commit_transaction()
#since we process from start to end, we have found our range:
db.close()

print('Completed')