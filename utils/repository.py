## utility class that contains often-needed queries

def get_images(treshold, minscore): 
    return f"""
    SELECT 
        angletag_predicts.*, 
        images.image_path, images.yolobox_top_left_x, 
        images.yolobox_top_left_y, images.yolobox_bottom_right_x, 
        images.yolobox_bottom_right_y,
        bintag_predicts.*,
        listings.brand
    FROM images
    JOIN bintag_predicts ON bintag_predicts.image_id = images.id
    JOIN angletag_predicts ON angletag_predicts.image_id = images.id
    JOIN listings ON listings.id = images.listing_id
    WHERE 
        (CASE WHEN model1_results >= {treshold} THEN 1 ELSE 0 END + 
        CASE WHEN model2_results >= {treshold} THEN 1 ELSE 0 END + 
        CASE WHEN model3_results >= {treshold} THEN 1 ELSE 0 END + 
        CASE WHEN model4_results >= {treshold} THEN 1 ELSE 0 END)
    >= {minscore};
    """