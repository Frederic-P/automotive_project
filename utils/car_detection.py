"""
Utilities used for detecting cars on an image using YOLO

"""
from PIL import Image 


def is_full_car(image_path, model):
    """Function to detect the largest car-tagged box in an image. 
    takes two arguments: 
    image_path == STR == FQ path to image.
    model == model == YOLO based model declared in main body. 
    """

    TRESHOLD_LOWER_RATIO = .20  #box should be minimum 20% of the image
    TRESHOLD_UPPER_RATIO = .95  #box should be smaller than 95% of the image
    MINIMUM_CONFIDENCE_SCORE = 0.50 #confidence score 
    #MINIMUM_DIFFERENCE = 0.1    #(Doesn't work as it should)if there are two boxes, the minimum surface area difference should be 10% biggest and 2nd biggest.
    img = Image.open(image_path) 
    # get width and height 
    width = img.width 
    height = img.height 
    # Run the YOLO model
    results = model(image_path, verbose = False)
    #print(width, height)
    hits = ['car', 'truck', 'vehicle']
    good_boxes = []
    for result in results:  # Iterate through each image's result
        for box, cls_id, confidence in zip(result.boxes.xyxy, result.boxes.cls, result.boxes.conf):
            class_name = result.names[int(cls_id)]  # Get class name
            # Check if the detected object is a car and if the bounding box is large enough
            if class_name in hits and confidence >= MINIMUM_CONFIDENCE_SCORE:
                # Here you can add logic to check the size of the box
                # For example, you could check if the box covers a significant portion of the image
                # Assuming box is in the format [x_min, y_min, x_max, y_max]
                box_width = (box[2] - box[0]) / width
                box_height = (box[3] - box[1]) /height
                box_surface = box_width * box_height
                if box_surface > TRESHOLD_LOWER_RATIO and box_surface <= TRESHOLD_UPPER_RATIO:
                    good_boxes.append([True, image_path,  box, confidence, box_surface])
    if len(good_boxes) == 0:
        return [False, image_path, False, False, 0]
    elif len(good_boxes) == 1:
        return good_boxes[0]
    else:
        #two or more: return biggest Experimenting with this code showed that YOLO has a tendency of nesting boxes.
        #so the same car has two boxes with marginal size differences. Too many false negatives because of this
        #do not use the minimum_difference, instead return the biggest box.
        sorted_boxes = sorted(good_boxes, key=lambda x: x[-1])
        return sorted_boxes[0]

        #calculate surface difference.
        #diff = (sorted_boxes[0][-1] - sorted_boxes[1][-1]) / (width*height)
        #if diff >= MINIMUM_DIFFERENCE:
        #    return sorted_boxes[0]
        #else:
        #    return [False, image_path, False, False, 0]
