import requests
import json
from time import sleep
from pathlib import Path
import numpy as np

class JrsObjectDetectionService:
    def __init__(self, server_name, server_port, model_name):
        self.server_name = server_name
        self.server_port = server_port
        self.batch_id = -1
        self.model_name = model_name

    def wait_for_completion(self, jobid):
        query = "http://" + self.server_name + ":" + self.server_port + "/job_status/"+jobid
        status = "ongoing"
        while status == "ongoing":
            response = requests.get(query)
            response_json = response.json()
            status = response_json["status"]
            if status == "ongoing":
                sleep(0.5)
            else:
                if status == "cancelled":
                    return None
                return response_json
        return None

    def poll_job_status(self, job_id):
        # poll status of the job
        error_msg = ""
        url = "http://" + self.server_name + ":" + self.server_port + "/job_status/" + str(job_id)
        data = {
            "jobid": job_id
        }

        while True:
            try:
                response = requests.get(url, data=data)
                if response.status_code == 200:
                    return_val = response.json()
                    if return_val is not None:

                        if return_val['status'] == 'completed':

                            return "completed", return_val, ""

                        elif return_val['status'] == 'cancelled':
                            error_msg = 'Error: ' + return_val['description']
                            return "cancelled", return_val, error_msg

                    # sleep for 0.5 sec
                    sleep(0.5)
                else:
                    try:
                        message_json = response.json()
                        if message_json is not None and 'description' in message_json:
                            error_msg = message_json['description']
                        else:
                            error_msg = f"Error at object detection service (HTTP error {response.status_code}, {response.text})!"
                        return "cancelled", None, error_msg
                    except Exception as e:
                        error_msg = f"Error at object detection service (HTTP error {response.status_code}, {response.text})!"
                        return "cancelled", None, error_msg
            except requests.exceptions.ConnectionError:
                error_msg = "Error: Connection to object detection service failed!"
                return "cancelled", None, error_msg


    def update_image(self, img_filename):

        query = "http://" + self.server_name + ":" + self.server_port + "/send_image"
        image_info = {
            "process_type": "extract",
        }
        image_info_json = json.dumps(image_info)

        batch_id = ''
        with open(img_filename, 'rb') as image_file:
            response = requests.post(query, files={'image': image_file}, data={'image_info': image_info_json})
            data = response.json()
            response_json = response.json()
            batch_id = data.get("batch_id")

        self.batch_id = batch_id


    def get_regions(self):

        url = "http://" + self.server_name + ":" + self.server_port + '/analyse_region'

        # Prepare the request payload
        request_body = {
            "batch_id": self.batch_id,
            "region": None,
            "class_selection_mode": 'all',
            "top_k": 1,
            "class_selection": None,
            "analyse_label": None,
            "use_background_mask": False,
            "model_id": self.model_name,
            "del_data": True
        }

        request_body_json = json.dumps(request_body)
        headers = {'Content-Type': 'application/json'}
        try:
            response = requests.post(url, data=request_body_json, headers=headers)
            # Check if the request was successful
            if response.status_code == 200:
                # Parse the JSON content
                data = response.json()
                data['image_patch_class_scores'] = np.array(data['image_patch_class_scores'])
                data['image_patch_top_class_indices'] = np.array(data['image_patch_top_class_indices'])

                return data, ""
            else:
                # Handle errors
                ret_error_msg = response.text
                error_msg = f"Error: Image region could not be analysed at the server: (HTTP error {response.status_code}, {ret_error_msg}!"
                return None, error_msg
        except requests.exceptions.ConnectionError:
            error_msg = "Error: Connection to object detection service failed!"
            return None, error_msg

    def get_score_image(self,do_not_care_1, do_not_care_2):
        analysis_result, error = self.get_regions()
        orig_image_patch_class_scores = analysis_result['image_patch_class_scores']
        return orig_image_patch_class_scores

    def detect_all_objects(self):
            url = "http://" + self.server_name + ":" + self.server_port + '/detect_objects'
            class_selection_mode="all"
            detect_supercategories=None
            del_uploaded_images=None
            images_dir = None
            with_embeddings = None
            request_body = {
                "batch_id": self.batch_id,
                "model_id": self.model_name,
                "class_selection_mode": class_selection_mode,
                "detect_supercategories": detect_supercategories,
                "del_data": del_uploaded_images,
                "validation_data": None,
                "with_embeddings": with_embeddings,
                "detection_threshold": 0.2,
                "ioa_filtering_threshold": 0.9,
                "top_k": 3
            }

            request_body_json = json.dumps(request_body)
            headers = {'Content-Type': 'application/json'}
            try:
                response = requests.post(url, data=request_body_json, headers=headers)
                if response.status_code == 200:
                    # poll status of the job
                    return_val = response.json()
                    job_id = return_val['jobid']
                    result = self.wait_for_completion(job_id)
                    result_list = result['result_list'][0]['features']

                    return result_list
                else:
                    try:
                        message_json = response.json()
                        if message_json is not None and 'description' in message_json:
                            error_msg = message_json['description']
                        else:
                            error_msg = f"Error at object detection service (HTTP error {response.status_code}, {response.text})!"
                    except Exception as e:
                        error_msg = f"Error at object detection service (HTTP error {response.status_code}, {response.text})!"
            except requests.exceptions.ConnectionError:
                error_msg = "Error: Connection to object detection service failed!"
            return [], error_msg

