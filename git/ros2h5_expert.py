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
import mediapy as media

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


def rosbag_to_array(filepath, episode, SAVE_PATH, index1, index2):
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
        action_index_list = []
        features_index_list = []
        SHOW_IMAGE = False
        START = False
        t1 = time.time()
        T = 0


        
        for connection, timestamp, rawdata in reader.messages():
            
            current_timestamp = timestamp - start_timestamp
            current_timestamp_sec = (current_timestamp / 1e9)
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
                #features = np.zeros((1, 1024))  # dummy features for testing
                features_buffer.append(features.reshape(-1))
                #features_buffer.append(image)

                features_index_buffer.append(current_time_index)

               
                # plot image
                if SHOW_IMAGE:
                    plt.imshow(image)
                    plt.axis('off')
                    plt.show()
            
            if len(action_index_buffer) > 8 and len(features_index_buffer) > 8:

                    START = True

            if  len(action_index_list) > 0 and len(features_index_list) > 0:
                time_diff = abs(action_index_list[0] - features_index_list[0])
                if time_diff > 0.05:
                    print('warning! time mis-match:', time_diff)
                    if action_index_list[0] < features_index_list[0]:
                        del action_index_list[0]
                        action_list.pop(0)
                    else:
                        del features_index_list[0]
                        features_list.pop(0)
       
                
            if START and len(action_index_buffer) > 1 and len(features_index_buffer) > 1:

                    action_index_list.append(action_index_buffer.pop(0))
                    features_index_list.append(features_index_buffer.pop(0))

                    features_list.append(features_buffer.pop(0))
                    action_list.append(action_buffer.pop(0))
                    
            






            if len(action_list) > 0 and len(features_list) > 0 and action_index_list[0] < index1[T]/4 and features_index_list[0] < index1[T]/4:
                del action_index_list[0]
                del features_index_list[0]
                del action_list[0]
                del features_list[0]


            if len(action_list) > 0 and len(features_list) > 0 and action_index_list[-1] > index2[T]/4 and features_index_list[-1] > index2[T]/4:
                print(f"the episode {episode} is saved with length:", len(action_list),'cost time:', time.time()-t1)
                t1 = time.time()
                min_number = min(len(action_list), len(features_list))
                action_array = np.array(action_list[0:min_number])
                features_array = np.array(features_list[0:min_number])
                del features_index_list[0:min_number]
                del action_index_list[0:min_number]
                del action_list[0:min_number]
                del features_list[0:min_number]
                with h5py.File(SAVE_PATH, "a") as f:


                    grp = f.create_group(f"episode_{episode}")
                    grp.create_dataset('features', data=features_array)
                    grp.create_dataset('actions', data=action_array)
                    episode += 1
                    T += 1


            if False and len(action_list) > 0 and len(features_list) > 0 and action_index_list[-1] > index2[T]/4 and features_index_list[-1] > index2[T]/4:


                media.write_video(
                    'expert_video'+f'_ep{episode}.mp4', 
                    features_list, 
                    fps=4)
                episode += 1
                del features_list[:]
                del action_list[:]
                del action_index_list[:]
                del features_index_list[:]
                T += 1

                


        







        
        print("onethe length for left action:", len(action_list))

    return action_array, features_array, episode



if __name__ == "__main__":
    episode = 1

    # expert

    PATH = {
        1: "/home/data/landrobot_data/11-10(4hz)/expet_data/subset/",
    }

    index1 = [1,47,80,134,180,226,252,295,362,415,462,530,571,605,664,723]
    index2 = np.array([34,69,119,165,214,239,263,352,400,453,518,558,593,649,710,765]) + 2



    SAVE_PATH = "S11_4Hz_expert_01.h5"

    for key in PATH.keys():
        bagpath = PATH[key]

        t1 = time.time()

        action_array, features_array, episode = rosbag_to_array(bagpath, episode=episode, SAVE_PATH=SAVE_PATH, index1=index1, index2=index2 )  
        gc.collect()

    
        
