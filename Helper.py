import numpy as np

def compute_intersection(rect1, rect2):

    #origin: lower left
    """
    Computes the intersection rectangle of two rectangles.
    Format: [xmin, ymin, xmax, ymax]
    """
    # Find the boundaries of the intersection box
    x_left = np.maximum(rect1[0], rect2[0])
    y_top = np.maximum(rect1[1], rect2[1])
    x_right = np.minimum(rect1[2], rect2[2])
    y_bottom = np.minimum(rect1[3], rect2[3])

    # Check if there is a valid intersection
    if x_right < x_left or y_top > y_bottom:
        return [0,0,0,0] # No intersection
        
    return np.array([x_left, y_top, x_right, y_bottom])


def get_most_probable_obj(detected_objects,wanted_label_id):
    highest_score = 0.0
    best_det_obj = None

    for det_obj in detected_objects:
        label = det_obj["class"]
        score = det_obj["confidence"]
        if label == wanted_label_id and score > highest_score:
            highest_score = score
            best_det_obj = det_obj

    return best_det_obj


def clip_reference_det_bbox(fromX, fromY, toX, toY, reference_det_fromX, reference_det_fromY, reference_det_toX, reference_det_toY):
        
    if fromX < reference_det_fromX:
        fromX = reference_det_fromX
    if toX > reference_det_toX:
        toX = reference_det_toX
    if fromY < reference_det_fromY:
        fromY = reference_det_fromY
    if toY > reference_det_toY:
        toY = reference_det_toY

    return fromX, fromY, toX, toY

def update_figure(f, y_axis_text, alg_type, label_of_interest, model_name):
    f.update_layout(
        title=dict(
            text = f"{alg_type}: {label_of_interest}<br><sup>Model:"+model_name+f"</sup><br>",
            xanchor = 'center',
            yanchor = 'top',
            x = 0.5,
            font=dict(
                family="Arial",
                size=48
            )
        ),
        xaxis=dict(
                title=dict(
                    text=f'Occlusion in % of object size'
                )
        ),
        yaxis=dict(
                title=dict(
                    text=y_axis_text
                )
        )
    )

