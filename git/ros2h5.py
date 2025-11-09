### using rosbag to read ros2 bag files and convert to h5 files

from pathlib import Path

from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_typestore

import numpy as np
import h5py
import matplotlib.pyplot as plt
import time
import gc
from transformers import AutoImageProcessor, AutoModel
from PIL import Image
import torch

def prepare_Dinov3():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    model_id = "facebook/dinov3-vitl16-pretrain-lvd1689m"
    processor = AutoImageProcessor.from_pretrained(model_id)
    model =   AutoModel.from_pretrained(model_id).to(device)
    return processor, model, device

def cal_latent(obs, processor, model, device):
    image = Image.fromarray(obs)
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad(): #
        outputs = model(**inputs)
    pooler_output = outputs.pooler_output
    return pooler_output.cpu().numpy()


def rosbag_to_array(filepath):
    bagpath = Path(filepath)
    t1 = time.time()
    typestore = get_typestore(Stores.ROS2_FOXY)
    processor, model, device = prepare_Dinov3()

    # Create reader instance and open for reading.
    with AnyReader([bagpath], default_typestore=typestore) as reader:

            # --- A. read（metadata.yaml） ---
        #print(f"Bag filepath: {reader.path}")
        print(f"start time: {reader.start_time}")
        print(f"end time: {reader.end_time}")
        print(f"message count: {reader.message_count}")
        print("-" * 30)

        start_timestamp = reader.start_time

        # List All Topic
        print("Topics and Types:")
        for connection in reader.connections:
            print(f"- {connection.topic}: {connection.msgtype}")
            print("-" * 30)

        
        current_timestamp = 0
        action_list = []
        image_list = []
        action_buffer = []
        image_buffer = []
        action_index_buffer = []
        features_buffer = []
        features_list = []
        features_index_buffer = []
        image_index_buffer = []
        SHOW_IMAGE = False
        START = False

   

        
        for connection, timestamp, rawdata in reader.messages():
            
            current_timestamp = timestamp - start_timestamp
            current_timestamp_sec = (current_timestamp / 1e8)
            current_time_index = current_timestamp_sec
            

            msg = reader.deserialize(rawdata, connection.msgtype)

            if connection.topic == '/OFFRL/action_state': 
                action = np.array([msg.linear.x, msg.linear.y, msg.angular.z])
                action_buffer.append(action)
                action_index_buffer.append(current_time_index)
     

            if connection.topic == '/OFFRL/camera_state':
                width = msg.width
                height = msg.height
                image = np.array(msg.data).reshape(height, width, -1)
                features = cal_latent(image, processor, model, device)
                features_buffer.append(features.reshape(-1))
                features_index_buffer.append(current_time_index)
               
                # plot image
                if SHOW_IMAGE:
                    plt.imshow(image)
                    plt.axis('off')
                    plt.show()
            
            if len(action_index_buffer) > 8 and len(features_index_buffer) > 8:

                    START = True
       
                
            if START and len(action_index_buffer) > 1 and len(features_index_buffer) > 1:
                   
                    del action_index_buffer[0]
                    del features_index_buffer[0]
                    action_list.append(action_buffer.pop(0))
                    features_list.append(features_buffer.pop(0))
                    
                
             
                
         
                    


        

        action_array = np.array(action_list)
        features_array = np.array(features_list)
        if len(action_array) < 20:
            print("too short episode")

    return action_array, features_array



if __name__ == "__main__":
    episode = 1
    PATH = {
        1: "/home/data/landrobot_data/data1104_pink_01/subset/",
        2: "/home/data/landrobot_data/data1104_pink_02/subset/",
        3: "/home/data/landrobot_data/data1104_pink_03/subset/",
        4: "/home/data/landrobot_data/data1104_pink_04/subset/",
        5: "/home/data/landrobot_data/data1104_pink_05/subset/",
    }

    # expert

    PATH = {
        1: "/home/data/landrobot_data/data_experts_1104/e01/subset/",
        2: "/home/data/landrobot_data/data_experts_1104/e02/subset/",
        3: "/home/data/landrobot_data/data_experts_1104/e03/subset/",
        4: "/home/data/landrobot_data/data_experts_1104/e04/subset/",
        5: "/home/data/landrobot_data/data_experts_1104/e05/subset/",
        6: "/home/data/landrobot_data/data_experts_1104/e06/subset/",
        7: "/home/data/landrobot_data/data_experts_1104/e07/subset/",
        8: "/home/data/landrobot_data/data_experts_1104/e08/subset/",
        9: "/home/data/landrobot_data/data_experts_1104/e09/subset/",
        10: "/home/data/landrobot_data/data_experts_1104/e10/subset/", 
        11: "/home/data/landrobot_data/data_experts_1104/e11/subset/",
        12: "/home/data/landrobot_data/data_experts_1104/e12/subset/",
        13: "/home/data/landrobot_data/data_experts_1104/e013(O)/subset/",
        14: "/home/data/landrobot_data/data_experts_1104/e14/subset/",
        15: "/home/data/landrobot_data/data_experts_1104/e15/subset/",
        16: "/home/data/landrobot_data/data_experts_1104/e16/subset/", 
        17: "/home/data/landrobot_data/data_experts_1104/e17/subset/",
        18: "/home/data/landrobot_data/data_experts_1104/e18/subset/",
        19: "/home/data/landrobot_data/data_experts_1104/e19/subset/",
        20: "/home/data/landrobot_data/data_experts_1104/e20/subset/",



        
    }

    for key in PATH.keys():
        episode = key
        bagpath = PATH[key]
        if bagpath == "/home/data/landrobot_data/data_experts_1104/e03/subset/":
            print("e03")

        t1 = time.time()

        action_array, features_array = rosbag_to_array(bagpath)  
        gc.collect()
    
        with h5py.File("S11_expert_raw_02.h5", "a") as f:

            grp = f.create_group(f"episode_{episode}")
            grp.create_dataset('features', data=features_array)
            grp.create_dataset('actions', data=action_array)

    
        
