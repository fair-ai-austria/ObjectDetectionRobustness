import argparse

import plotly.graph_objects as go
import plotly.express as px

from PIL import Image, ImageDraw, ImageFont
from io import BytesIO
import numpy as np

from os import listdir
from os.path import isfile, join
from pathlib import Path

from JrsObjectDetectionService import JrsObjectDetectionService
from CustomObjectDetectionService import CustomObjectDetectionService
from Helper import compute_intersection, get_most_probable_obj, clip_reference_det_bbox, update_figure

parser = argparse.ArgumentParser(description="Arguments for Evaluation Script")

parser.add_argument("--jrs_server_name", type=str, help="Server Name or IP with running Object Detection Service")
parser.add_argument("--jrs_server_port", type=str, help="Server Port of running Object Detection Service")
parser.add_argument("--jrs_model_name", type=str, help="Model Name to be used by Object Detection Service")
parser.add_argument("-label_of_interest", type=str, help="Object Name (Label of Interest) to be detected by Object Detection Service")
parser.add_argument("-enable_region_approach", action="store_true", help="Occlusion Strategy: Region based approach")
parser.add_argument("-enable_dino_approach", action="store_true", help="Occlusion Strategy: Dino Patch approach")
parser.add_argument("-enable_l2r_approach", action="store_true", help="Occlusion Strategy: Left to Right")
parser.add_argument("-enable_global_summary", action="store_true", help="Gives an overview of object template set robustness, only one of Dino, L2R or Region must be enabled")
parser.add_argument("-input_path_object_templates",type=str, help="Reads all jpg files and tries to detect Label_Of_Interest in it")
parser.add_argument("-output_path", type=str, help="Reads all jpg files and tries to detect Label_Of_Interest in it")

args = parser.parse_args()

model_name=''#used as header in diagram
if args.jrs_server_name is not None and args.jrs_server_port is not None and args.jrs_model_name is not None: 
    model_name = args.jrs_model_name
    ods = JrsObjectDetectionService(args.jrs_server_name, args.jrs_server_port, args.jrs_model_name)
else:
    ods = CustomObjectDetectionService() #to be implemented by 3rd party

label_of_interest = args.label_of_interest #object name to be detected

enable_region_approach = args.enable_region_approach
enable_dino_patch_approach = args.enable_dino_approach
enable_l2r_approach = args.enable_l2r_approach

enable_global_summary = args.enable_global_summary #diagram: number of templates over occlusion (this option gives a robustness overview of a whole template set)

global_fig = go.Figure()

region_fig = go.Figure()
dino_fig = go.Figure()
l2r_fig = go.Figure()

alg_type = ""

global_occlusion_in_percent = []
global_occlusion_in_percent_last_value = [] #helper

input_path = Path(f"{args.input_path_object_templates}/{label_of_interest}")
input_files = [] #files will be collected later below

output_path_base = args.output_path+label_of_interest+"/"


#get list of jpeg images
input_files = [
    f for f in input_path.iterdir()
    if f.is_file() and f.suffix.lower() in (".jpg", ".jpeg")
]

for input_file in input_files:
    input_image_file_name = str(input_file)
    file_id = input_file.stem

    output_path = output_path_base + file_id + "/"

    output_path_region = output_path + "region/"
    output_path_dino = output_path + "dino_patch/"
    output_path_l2r = output_path + "l2r/"

    intersection_threshold = 0.3


    wanted_label_id = -1


    image_file_name_tmp = input_image_file_name

    Path(output_path).mkdir(parents=True, exist_ok=True)

    original_input_image = Image.open(input_image_file_name)

    #get area of reference object for later restricting inpainting to this area
    reference_det_bbox = None

    ods.update_image(input_image_file_name)
    detected_objects = ods.detect_all_objects()

    reference_det_obj = get_most_probable_obj(detected_objects, label_of_interest)


    if ( reference_det_obj != None):
        reference_det_bbox = reference_det_obj["region"]
    else:
        termination_str=f"Initial detection: No object of type {label_of_interest} found, ignoring object template {input_image_file_name}."
        print(termination_str)
        continue

    reference_det_bbox_size = (reference_det_bbox[2]-reference_det_bbox[0])*(reference_det_bbox[3]-reference_det_bbox[1])

    reference_det_fromX = reference_det_bbox[0]
    reference_det_fromY = reference_det_bbox[1]
    reference_det_toX = reference_det_bbox[2]
    reference_det_toY = reference_det_bbox[3]

    orig_image_patch_class_scores = ods.get_score_image(input_image_file_name, label_of_interest)

    orig_image_patch_class_scores = orig_image_patch_class_scores.squeeze()
    working_image_patch_class_scores = orig_image_patch_class_scores.copy()

    #scale scores to [0,255]
    max_score = orig_image_patch_class_scores.max()
    scale_factor = 255.0 / max_score
    scaled_class_scores_float = np.multiply(orig_image_patch_class_scores, scale_factor) 
    scaled_class_scores_int = scaled_class_scores_float.astype(np.uint8)

    scaled_class_scores_img = Image.fromarray(scaled_class_scores_int)
    scaled_class_scores_img = scaled_class_scores_img.resize((original_input_image.width, original_input_image.height),Image.NEAREST)

    #fill bounding box of reference detection to 1
    reference_det_mask = np.zeros_like(scaled_class_scores_img)
    reference_det_mask[reference_det_fromY:reference_det_toY,reference_det_fromX:reference_det_toX]=1
    reference_det_mask = np.astype(reference_det_mask,np.uint8)

    if enable_region_approach:

        alg_type = "Region mask" # used for diagram title

        Path(output_path_region).mkdir(parents=True, exist_ok=True)

        occlusion_in_percent = []
        detection_scores = []
        occlusion_iteration = 0
        quantization_steps = 3
        stop_region_growing_when_reference_det_is_covered_by_more_than = 0.9


        #quantize score levels
        scaled_class_scores_tmp_int = np.array(scaled_class_scores_img)
        scaled_class_scores_tmp_int = np.divide( scaled_class_scores_tmp_int, quantization_steps)
        scaled_class_scores_tmp_int = np.astype(scaled_class_scores_tmp_int, np.uint8)
        scaled_class_scores_tmp_int = np.multiply(scaled_class_scores_tmp_int, quantization_steps)
        scaled_class_scores_tmp_int = np.astype(scaled_class_scores_tmp_int, np.uint8)

        scaled_class_scores_tmp_int_3ch = np.stack([scaled_class_scores_tmp_int,scaled_class_scores_tmp_int,scaled_class_scores_tmp_int], axis=2)
        quantized_scores_img = Image.fromarray(scaled_class_scores_tmp_int_3ch)

        #set object bounding box to 0, set background to 1
        inv_reference_det_mask = np.where(reference_det_mask>0,0,1)
        inv_reference_det_mask_ch3 = np.stack([inv_reference_det_mask, inv_reference_det_mask, inv_reference_det_mask], axis=2)

        termination_str="" #provide the user with some textual feedback

        ref_object_occluded_in_percent = 0
        last_ref_object_occluded_in_percent = 0
        
        gray_levels = list(range(255,0,-quantization_steps))
        #loop from bright gray level (high scores) to dark gray level (low scores)
        for gray_level in gray_levels:
            
            #start loop always with clean original_input_image
            working_input_image_tmp_1 = np.multiply(original_input_image, inv_reference_det_mask_ch3)
            working_input_image_tmp_1 = np.astype(working_input_image_tmp_1,np.uint8)

            #set all pixels with scores >= gray level to 0, set all other pixels to 1 (helps to remove high score pixel values later)
            scores_mask_global = np.where(scaled_class_scores_tmp_int>=gray_level, 0, 1 )
            scores_mask_global = np.astype(scores_mask_global, np.uint8)

            #get only pixels within bounding box of detected reference object with low scores (remove pixels with high scores for object detection to test object template robustness)
            low_scores_obj_mask = np.multiply(scores_mask_global, reference_det_mask)
            low_scores_obj_mask = np.astype(low_scores_obj_mask,np.uint8)

            #prepare for 3 channel image
            low_scores_obj_mask_ch3 = np.stack([low_scores_obj_mask, low_scores_obj_mask, low_scores_obj_mask], axis=2)

            #mask in object gray scale values where scores are low (high score pixels are masked out now)
            working_input_image_tmp_2 = np.multiply( low_scores_obj_mask_ch3, original_input_image )
            working_input_image_tmp_2 = np.astype(working_input_image_tmp_2,np.uint8)

            #add background 
            working_input_image_tmp_3 = np.add(working_input_image_tmp_1, working_input_image_tmp_2)
            working_input_image_tmp_3 = np.astype(working_input_image_tmp_3,np.uint8)

            #number of low score pixels
            low_score_pixel_count = np.sum(low_scores_obj_mask)
            high_score_pixel_count = reference_det_bbox_size - low_score_pixel_count
            high_score_pixel_count_ratio = high_score_pixel_count / reference_det_bbox_size
            ref_object_occluded_in_percent = high_score_pixel_count_ratio * 100.0

            if ref_object_occluded_in_percent != last_ref_object_occluded_in_percent:

                if high_score_pixel_count_ratio > 0:
                    modified_input_image = Image.fromarray(working_input_image_tmp_3)

                    image_file_name_tmp = output_path_region+f"region_iter_{occlusion_iteration}_{gray_level}.png" 

                    modified_input_image.save(image_file_name_tmp) #save file because update_image reads it from there

                    ods.update_image(image_file_name_tmp)
                    #at first try to detect an object roughly (bounding box wanted)
                    detected_objects = ods.detect_all_objects()

                    detection_score = -1.0
                    wanted_obj = None

                    # look for object 'label_of_interest' with highest intersection ratio wrt reference object
                    last_intersection_ratio = 0.0
                    for det_obj in detected_objects:

                        label_inpainted = det_obj["class"]
                        if label_inpainted == label_of_interest:
                            det_bbox_inpainted = det_obj["region"]

                            #still same object detected?
                            intersection = compute_intersection(reference_det_bbox, det_bbox_inpainted)
                            intersection_size = (intersection[3]-intersection[1])*(intersection[2]-intersection[0])
                            intersection_ratio = intersection_size / reference_det_bbox_size
                            if intersection_ratio >= last_intersection_ratio:
                                wanted_obj = det_obj
                                detection_score = wanted_obj["confidence"]
                                last_intersection_ratio = intersection_ratio

                    if last_intersection_ratio < intersection_threshold:
                        termination_str = f"{alg_type} Finished: Intersection of detected object with reference object too small, I={last_intersection_ratio}, minIntersection={intersection_threshold}"
                        break

                    detection_scores.append(detection_score)

                    if wanted_obj == None:
                        termination_str = f"{alg_type} Finished: No {label_of_interest} found."
                        print(termination_str)
                        break

                    det_bbox = wanted_obj["region"]

                    #draw bounding box and save for debug
                    image_with_detection = Image.fromarray(working_input_image_tmp_3)
                    draw = ImageDraw.Draw(image_with_detection)
                    draw.rectangle(reference_det_bbox, outline="yellow", width=3)#originally detected object
                    draw.rectangle(det_bbox, outline="blue", width=3)

                    font = ImageFont.truetype('arial.ttf', 20) 
                    wanted_obj_label = wanted_obj["class"]
                    draw.text((det_bbox[0],det_bbox[1]),f"{wanted_obj_label}: {detection_score}",font=font, fill="blue")

                    image_with_detection.save(output_path_region+f"region_iter_{occlusion_iteration}_{gray_level}_detection.png")

                    print(f"{alg_type}: Iteration {occlusion_iteration}, (score) grey_level>={gray_level}, occ={ref_object_occluded_in_percent}")

                    occlusion_in_percent.append(ref_object_occluded_in_percent)

            last_ref_object_occluded_in_percent = ref_object_occluded_in_percent
            occlusion_iteration = occlusion_iteration + 1

            if high_score_pixel_count_ratio > stop_region_growing_when_reference_det_is_covered_by_more_than:
                termination_str = f"{alg_type}: Terminate, region occupies {high_score_pixel_count_ratio} (fraction) of reference object, max. occupation={stop_region_growing_when_reference_det_is_covered_by_more_than}."
                print(termination_str)
                break

        region_fig.add_trace(go.Scatter(x=occlusion_in_percent, y=detection_scores, name =f'Detection Score ({file_id})', mode='lines'))

        y_axis_text = "Detection score"

        update_figure(region_fig, y_axis_text, alg_type, label_of_interest, model_name)
        region_fig.write_html(output_path_base + f"DetectionScores_{alg_type}_{label_of_interest}_{file_id}.html")

    if enable_dino_patch_approach:

        alg_type = "Dino Patch"

        Path(output_path_dino).mkdir(parents=True, exist_ok=True)

        occlusion_iteration = 0
        occlusion_in_percent = []
        detection_scores = []

        termination_str =""#provide the user with some textual feedback

        #score/object id images returned from object detection service have different size, e.g. about 50x50
        #Masking out affects the whole block of pixels which correspond to a single pixel in the 50x50 image
        #Compute scale factors between different resolutions
        scaleX_to_dino = orig_image_patch_class_scores.shape[1] / original_input_image.width  
        scaleY_to_dino = orig_image_patch_class_scores.shape[0] / original_input_image.height

        #reference object position and size in "dino resolution" (e.g. 50x50)
        fromX_dino = int(reference_det_fromX * scaleX_to_dino)
        fromY_dino = int(reference_det_fromY * scaleY_to_dino)
        toX_dino = int(reference_det_toX * scaleX_to_dino)
        toY_dino = int(reference_det_toY * scaleY_to_dino)

        #set reference object bounding box to 1 in "dino resolution"
        reference_det_mask_dino = np.zeros_like(working_image_patch_class_scores)
        reference_det_mask_dino[fromY_dino:toY_dino,fromX_dino:toX_dino]=1

        working_image_patch_class_scores = np.multiply(working_image_patch_class_scores, reference_det_mask_dino)

        #generate inpainting indices (score maxima in descending sort order): list of pixel locations to be masked out (descending sort order by score value)
        runs = working_image_patch_class_scores.shape[0]*working_image_patch_class_scores.shape[1]
        runs = int(runs)
        occluding_pos_list=[]
        orig_max_value = working_image_patch_class_scores.max()
        
        #iterate over whole image
        for run in range(runs):
            max_value = working_image_patch_class_scores.max()
            if max_value > 0.0:
                max_score_index_row,max_score_index_column = np.unravel_index(working_image_patch_class_scores.argmax(), working_image_patch_class_scores.shape)
                occluding_pos_list.append((max_score_index_row,max_score_index_column,max_value))

                working_image_patch_class_scores[max_score_index_row,max_score_index_column]=0.0;
            else:
                #no maximum found any more
                break

        
        ods.update_image(input_image_file_name)

        #occlude..
        ref_object_occluded_in_percent = 0
        working_input_image = np.array(original_input_image.copy()) #start with clean input image
        masked_image_filename = input_image_file_name

        for occluding_pos in occluding_pos_list:

            #at first try to detect an object roughly (bounding box wanted)
            detected_objects = ods.detect_all_objects()

            detection_score = -1.0
            wanted_obj = None

            # look for wanted object of type 'label_of_interest'
            last_intersection_ratio = 0.0
            intersection_ratio = 0.0
            for det_obj in detected_objects:

                label_inpainted = det_obj["class"]
                if label_inpainted == label_of_interest:
                    det_bbox_inpainted = det_obj["region"]

                    #Still same object detected? Compute intersection with reference detection, there must be a minimum intersection, otherwise we are not interested in it
                    intersection = compute_intersection(reference_det_bbox, det_bbox_inpainted)
                    intersection_size = (intersection[3]-intersection[1])*(intersection[2]-intersection[0])
                    intersection_ratio = intersection_size / reference_det_bbox_size
                    if intersection_ratio >= last_intersection_ratio:
                        wanted_obj = det_obj
                        detection_score = wanted_obj["confidence"]
                        last_intersection_ratio = intersection_ratio

            if last_intersection_ratio >= intersection_threshold:
                detection_scores.append(detection_score)
            else:
                termination_str = f"{alg_type} Finished: Intersection of detected object with reference object too small, Intersection={last_intersection_ratio}, min. Intersection={intersection_threshold}"
                print(termination_str)
                break

            if wanted_obj == None:
                termination_str = f"{alg_type}: Terminating. No {label_of_interest} found."
                print(termination_str)
                break

            det_bbox = wanted_obj["region"]
    
            #occlude at max score location
            scaleX = original_input_image.width / orig_image_patch_class_scores.shape[1] 
            scaleY = original_input_image.height / orig_image_patch_class_scores.shape[0] 

            max_score_index_row = occluding_pos[0]
            max_score_index_column = occluding_pos[1]

            #scale to original input image size
            fromX = int(max_score_index_column * scaleX)
            toX = int(( max_score_index_column + 1 ) * scaleX)
            fromY = int(max_score_index_row * scaleY)
            toY = int(( max_score_index_row + 1 ) * scaleY)

            #prevents occlusion of more than 100%
            fromX, fromY, toX, toY = clip_reference_det_bbox(fromX, fromY, toX, toY, reference_det_fromX, reference_det_fromY, reference_det_toX, reference_det_toY)

            #mask them out! Pixels with max score are set to black
            working_input_image[fromY:toY,fromX:toX,:]=0

            save_working_input_image = Image.fromarray(working_input_image)
            #save_inpainted_image = save_inpainted_image.resize((original_input_image.width,original_input_image.height))
            masked_image_filename = output_path_dino + f"dino_patch_iter_{occlusion_iteration}.png"
            save_working_input_image.save(masked_image_filename)#this destroys the original image because it is inpainted with black bars

            #draw bounding box and save for debug
            save_working_input_image = save_working_input_image.copy() #make deep copy, avoid modification of working_input_image by drawing in it
            draw = ImageDraw.Draw(save_working_input_image)
            draw.rectangle(reference_det_bbox, outline="yellow", width=3)#original detected object
            draw.rectangle(det_bbox, outline="blue", width=3)#re-detection although object only partially visible (pixels masked out)

            font = ImageFont.truetype('arial.ttf', 20) 
            wanted_obj_label = wanted_obj["class"]
            draw.text((det_bbox[0],det_bbox[1]),f"{wanted_obj_label}: {detection_score}",font=font, fill="blue")

            save_working_input_image.save(output_path_dino+f"dino_patch_iter_{occlusion_iteration}_detection.png")

            ods.update_image(masked_image_filename)

            max_value = occluding_pos[2]
            print(f"{alg_type}: Iteration {occlusion_iteration}/{len(occluding_pos_list)}: detection score {detection_score}, suppressed max score {max_value} at ({max_score_index_column},{max_score_index_row}), occ={ref_object_occluded_in_percent}")

            patch_size = (toX-fromX)*(toY-fromY)
            occluded_in_percent = patch_size / reference_det_bbox_size * 100.0
            ref_object_occluded_in_percent = ref_object_occluded_in_percent + occluded_in_percent
            occlusion_in_percent.append(ref_object_occluded_in_percent)
            occlusion_iteration = occlusion_iteration + 1


        dino_fig.add_trace(go.Scatter(x=occlusion_in_percent, y=detection_scores, name =f'Detection Score ({file_id})', mode='lines'))

        y_axis_text = "Detection score"
        update_figure(dino_fig, y_axis_text, alg_type, label_of_interest, model_name)
        dino_fig.write_html(output_path_base + f"DetectionScores_{alg_type}_{label_of_interest}_{file_id}.html")

    if enable_l2r_approach:

        alg_type = "Left to Right"

        Path(output_path_l2r).mkdir(parents=True, exist_ok=True)

        occlude_iteration = 0
        occlude_iterations = []
        occlusion_in_percent = []
        detection_scores = []

        ods.update_image(input_image_file_name) #init image on server

        termination_str =""
        ref_object_occluded_in_percent = 0
        working_input_image = np.array(original_input_image.copy())
        masked_image_filename = input_image_file_name

        #reference_det values have been inited in 'global section' above, no need for re-detection
        columns = range(reference_det_fromX, reference_det_toX)
        
        #iterate from left to right column of reference object bounding box
        for column in columns:

            #at first try to detect an object roughly (bounding box wanted)
            detected_objects = ods.detect_all_objects()

            detection_score = -1.0
            wanted_obj = None

            # look for re-detection of reference object of type 'label_of_interest'
            last_intersection_ratio = 0.0
            for det_obj in detected_objects:

                label_inpainted = det_obj["class"]
                if label_inpainted == label_of_interest:
                    det_bbox_inpainted = det_obj["region"]

                    #still reference object detected?
                    intersection = compute_intersection(reference_det_bbox, det_bbox_inpainted)
                    intersection_size = (intersection[3]-intersection[1])*(intersection[2]-intersection[0])
                    intersection_ratio = intersection_size / reference_det_bbox_size
                    if intersection_ratio >= last_intersection_ratio:
                        wanted_obj = det_obj
                        detection_score = wanted_obj["confidence"]
                        last_intersection_ratio = intersection_ratio

        
            if last_intersection_ratio >= intersection_threshold:
                detection_scores.append(detection_score)


            if wanted_obj == None:
                termination_str = f"{alg_type} Finished: No {label_of_interest} found."
                print(termination_str)
                break

            det_bbox = wanted_obj["region"]

            #occlude
            working_input_image[reference_det_fromY:reference_det_toY,column,:]=0

            #occlusion of reference object
            occlusion = (reference_det_toY-reference_det_fromY) / reference_det_bbox_size * 100.0
            ref_object_occluded_in_percent = ref_object_occluded_in_percent + occlusion

            save_inpainted_image = Image.fromarray(working_input_image)
            #save_inpainted_image = save_inpainted_image.resize((original_input_image.width,original_input_image.height))
            masked_image_filename = output_path_l2r + f"occlusion_iter_{occlude_iteration}.png"
            save_inpainted_image.save(masked_image_filename)#this destroys the original image because it is inpainted with black bars

            #draw detection and save for debug
            save_inpainted_image = save_inpainted_image.copy()
            draw = ImageDraw.Draw(save_inpainted_image)
            draw.rectangle(reference_det_bbox, outline="yellow", width=3)#original detected object
            draw.rectangle(det_bbox, outline="blue", width=3)

            font = ImageFont.truetype('arial.ttf', 20) 
            wanted_obj_label = wanted_obj["class"]
            draw.text((det_bbox[0],det_bbox[1]),f"{wanted_obj_label}: {detection_score}",font=font, fill="blue")

            #update image on server
            ods.update_image(masked_image_filename)


            print(f"{alg_type}: Iteration {occlude_iteration}/{len(columns)}, {file_id}: detection score {detection_score}")
            occlusion_in_percent.append(ref_object_occluded_in_percent)
            occlude_iteration = occlude_iteration + 1

            if last_intersection_ratio < intersection_threshold:
                
                termination_str = f"{alg_type} Finished: Intersection of detected object with reference object too small, I={last_intersection_ratio}, minIntersection={intersection_threshold}, occ={ref_object_occluded_in_percent}"
                print(termination_str)
                break


        l2r_fig.add_trace(go.Scatter(x=occlusion_in_percent, y=detection_scores, name =f'Detection Score ({file_id})', mode='lines'))

            
        y_axis_text = "Detection score"
        update_figure(l2r_fig, y_axis_text, alg_type, label_of_interest, model_name)
        l2r_fig.write_html(output_path_base + f"DetectionScores_{alg_type}_{label_of_interest}_{file_id}.html")

    #this code section is valid for each alg_type, but: Only one alg_type must be selected!
    # accumulate all occlusion_in_percent lines
    global_occlusion_in_percent = list(set(global_occlusion_in_percent+occlusion_in_percent))# 'set' allows no duplicates
    
    #take last elem of occlusion_in_percent and add to global_occlusion_in_percent_last_value, the last value is assumed to be unique and serves as a marker for the specific line (object template) in the line chart
    if len(occlusion_in_percent)>0:
        last_elem = occlusion_in_percent[-1]
        global_occlusion_in_percent_last_value.append(last_elem)


if enable_region_approach:
    region_fig.show()

if enable_dino_patch_approach:
    dino_fig.show()

if enable_l2r_approach:
    l2r_fig.show()


if enable_global_summary:
    #display number of templates over occlusion in percent: Compress the various score over occ curves to a single number of templates over occ curve.
    #Rough outline of how number of active templates line is computed: We go along the x-axis (occlusion) in the diagram starting at 0, at first - no occlusion - all templates are active.
    #Each time a last elem marker is detected the number of active templates is reduced by 1.
   
    global_occlusion_in_percent.sort()
    number_of_active_templates_val = len(global_occlusion_in_percent_last_value) # number of last values is equal to number of lines / object templates 
    number_of_active_templates_list = []
    for occlusion in global_occlusion_in_percent:
        number_of_active_templates_list.append(number_of_active_templates_val)
        if occlusion in global_occlusion_in_percent_last_value:
            number_of_active_templates_val = number_of_active_templates_val - 1

    if len(number_of_active_templates_list)>0:
        number_of_active_templates = number_of_active_templates_list[0]
        number_of_active_templates_list_normed = np.divide(number_of_active_templates_list , number_of_active_templates)

        global_fig.add_trace(go.Scatter(x=global_occlusion_in_percent, y=number_of_active_templates_list_normed, name =f'Detection Score ({file_id})', mode='lines'))

        global_fig.update_layout(
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
                        text="Fraction of reliable templates"
                    )
            )
            ,yaxis_range=[0,1]
        )

        global_fig.show()
        global_fig.write_html(output_path_base + f"Detectionscores_{alg_type}_{label_of_interest}_reliable_templates_summary.html")
