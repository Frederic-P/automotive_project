"""
    headless script developped based on findings gained in detect_car.ipynb
    goal of this script is to find the largest YOLObox that has a label 'car', 'truck' or 'vehicle'
    and save the box coords in the SQL database.

    Under imports you can set parameters for batch processing. 

"""
#basic modules
import os
from ultralytics import YOLO
from tqdm import tqdm
import math
##issue on UNIX systems with YOLO: https://github.com/ultralytics/yolov5/issues/1298

import sys
sys.path.append('utils')
import config_handling as conf
from database import Database
import car_detection as cd

##Batch processing config: 
batch_size = 250

if os.name == 'posix':
    print('A unix system was detected, please follow these steps to patch a known UNIX bug:')
    print('https://askubuntu.com/questions/1299255/how-can-i-solve-no-module-named-lzma')
    input('To continue press enter')

# Load the YOLO model
yolomodel = YOLO('models/yolo/yolov8n.pt')  # YOLOv8 nano for speed, or 'yolov8s.pt' for more accuracy

# Load the project configuration: 
config = conf.read_config('config/automotive.conf.ini')
imdir = config['settings']['image_directory']

#connect to database: 
connection_type = config['settings']['connection']
connection_type
user = config[connection_type]['user']
pw = config[connection_type]['pw']
host = config[connection_type]['host']
db = config[connection_type]['db']
port = config[connection_type].getint('port')
db = Database(host,
              port,
              user,
              pw,
              db
              )
db.connect()

#get basedir: 
basedir = config['settings']['image_directory']
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
for id in tqdm(range(imids[0]['minrange'], imids[0]['maxrange'], batch_size)):
    batch = "SELECT id, image_path FROM images WHERE id BETWEEN %s AND %s AND processed = 0;"
    batch_args = [id, id+batch_size]
    batch_images = db.execute_query(batch, batch_args)
    db.start_transaction()
    for image in batch_images:
        pk_id = image['id']
        path = os.path.join(basedir, image['image_path'].replace('\\', '/'))
        if not os.path.exists(path):
            with open('logging/automated_yolo_tagger.txt', 'a+') as f:
                f.write(path, 'does not exists', '\n')
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