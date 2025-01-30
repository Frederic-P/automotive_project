from flask import Flask, render_template, send_file, request, redirect, url_for, session
import os
import json
import sys
sys.path.append('../utils')
import config_handling as conf
from database import Database
from file_io import path_handler

# Connect to database
config = conf.read_config('../config/automotive.conf.ini')
config.read('config.ini')
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

#image directory: 
basedir = config['settings']['image_directory']
app = Flask(__name__)
app.secret_key = os.urandom(24)

def load_app_settings():
    result_dict = {}
    with open('flask_settings.txt', 'r') as file:
        for line in file:
            # Strip whitespace and check if line is not empty
            line = line.strip()
            if line:
                # Split the line into key and value
                key, value = line.split(':', 1)  # Split only on the first colon
                # Strip whitespace from key and value
                result_dict[key] = value
    return result_dict

# Load credentials from secrets.txt
def load_credentials():
    with open('secret.txt', 'r') as f:
        return f.read().strip().split(':')

username, password = load_credentials()
app_settings = load_app_settings()


def pick_random_image(): 
    """picks a random image matching the country restraint.
        even after indexing it was faster to perform two small queries over 1 query using join logic. 
        ##some listings have no image, so we use a while loop - good enough (still faster than relying on joins.)
    """
    image = []
    while len(image) == 0: 
        ##phase 1 pick random listing:
        listing_query = "SELECT * FROM listings WHERE countrycode  = %s ORDER BY RAND() LIMIT 1;"
        listing_param = [app_settings['countrycode'].strip()]
        listing = db.execute_query(listing_query, listing_param)
        # print(listing)
        #Phase 2: use the listing id to get a random image (that's not yet tagged)
        listing_id = listing[0]['id']
        image_query = """SELECT images.* 
            FROM images 
            WHERE images.listing_id = %s 
            AND images.id NOT IN (SELECT image_id FROM angletags WHERE manual_annotation = 1) 
            ORDER BY RAND() 
            LIMIT 1;"""
        image_param = [listing_id]
        image = db.execute_query(image_query, image_param)
    #return first row (you always have one row by definition; unless you missconfigure the countrycode, but if you manage that, well...)
    return [listing[0], image[0]]


def get_stats(): 
    """fetches the counter for each angle. """
    query = "SELECT angle, COUNT(*) as count FROM angletags GROUP BY angle ORDER BY count DESC"
    angle_data =  db.execute_query(query, [])
    angle_count_dict = {item['angle']: item['count'] for item in angle_data}
    return angle_count_dict


@app.route('/')
def index():
    if 'logged_in' in session:
        listing, image = pick_random_image()
        
        brand = listing['brand']
        model = listing['model']
        brand_autoscout = listing['make_name_autoscout']
        model_autoscout = listing['model_name_autoscout']
        year = listing['year']
        shell = listing['shelltype']
        doors = listing['doorcount']
        car_name = brand + ' ' + model
        autoscout_name = brand_autoscout+ ' ' + model_autoscout
        full_shell = shell + ' - ' + str(doors)
        
        relative_path = image['image_path']
        yolo_topleft_x = json.dumps(image['yolobox_top_left_x'])
        yolo_topleft_y = json.dumps(image['yolobox_top_left_y'])
        yolo_bottomright_x = json.dumps(image['yolobox_bottom_right_x'])
        yolo_bottomright_y = json.dumps(image['yolobox_bottom_right_y'])
        image_path = path_handler(basedir, relative_path)
        confidence = 0
        if image['confidence'] is not None:
            confidence = image['confidence']
        yolo_certainty = round(confidence * 100, 3)

        # Use a dynamic route to serve the image
        print(image_path)
        print(os.path.exists(image_path))
        imadr = url_for('serve_image', filepath=image_path)
        stats = get_stats()
        return render_template(
            'index.html',
            name=car_name,
            autoscout_name = autoscout_name, 
            year=year,
            shell=full_shell,
            certainty=yolo_certainty,
            imadr=imadr, 
            x1 = yolo_topleft_x,
            y1 = yolo_topleft_y, 
            x2 = yolo_bottomright_x,
            y2 = yolo_bottomright_y,
            image_id = image['id'], 
            stats = stats
        )
    return redirect(url_for('login'))

# Route to dynamically serve images from any path
@app.route('/serve_image/<path:filepath>')
def serve_image(filepath):
    #todo: make this check better with decent UNIX detection as in YOLO tagger. 
    if filepath.startswith('home'):
        filepath = '/'+filepath
    print(filepath)
    if os.path.exists(filepath):
        return send_file(filepath)

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        if request.form['username'] == username and request.form['password'] == password:
            session['logged_in'] = True
            return redirect(url_for('index'))
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('logged_in', None)
    return redirect(url_for('login'))

@app.route('/angle', methods=['POST'])
def angle():
    angle_value = request.form['angle']
    image_id = request.form['imid']
    query_angle = "INSERT INTO angletags (image_id, angle, manual_annotation) VALUES(%s, %s, %s)"
    query_data = [image_id, angle_value, 1]
    db.start_transaction()
    db.execute_query(query_angle, query_data)
    db.commit_transaction()
    return redirect(url_for('index'))

@app.route('/undo', methods=['GET'])
def undo():
    if 'logged_in' not in session:
        return redirect(url_for('index'))

    delete_last = """DELETE FROM angletags WHERE pk = (SELECT pk FROM (SELECT MAX(pk) as pk FROM angletags) AS temp);"""
    try:
        db.start_transaction()
        db.execute_query(delete_last)
        db.commit_transaction()
    except Exception as e:
        db.rollback_transaction()
        print(f"Error during undo operation: {e}")
        return "An error occurred while undoing the action."
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)