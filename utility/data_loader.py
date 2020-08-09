import numpy as np
from utility.feature_extract import ModelExtractFaceFeature
from PIL import Image
from glob import glob
import os
import torch

def data_load():
    model_eff = ModelExtractFaceFeature()
    img_dir = "./data/test_image4tani_2"
    img_path_list = glob(img_dir + '/*')

    img_path_list_length = len(img_path_list)
    img_save_flag = False
    img_crop_list = list()    
    feature_list = list()

    #dataの格納
    for idx, img_path in enumerate(img_path_list):
        if img_save_flag:
            img_name = os.path.basename(img_path)
            img_save_dir = "./data/result/"
            img_save_path = img_save_dir + img_name
        else:
            img_save_path = None            

        try:
            img = Image.open(img_path)
            img_cropped = model_eff.trim_img(img.resize((160, 160)), model_eff.trim_face_model, img_path=img_save_path)
            img_cropped = img_cropped.to('cpu').detach().numpy().copy()
            img_crop_list.append(img_cropped)
                
        except:
            print("Error_img_cropped --> ", img_cropped.shape)
            continue

    #推論する
    img_crop_arr = torch.from_numpy(np.array(img_crop_list).astype(np.float32)).clone()
    feature = model_eff.inference(img_crop_arr, model_eff.extract_feature_model)
    # feature = model_eff.inference(img_cropped, model_eff.extract_feature_model)
    feature_numpy = feature.to('cpu').detach().numpy().copy()
    feature_list.append(feature_numpy)
    
    return np.array(feature_list), img_path_list


if __name__ == "__main__":
    pass
