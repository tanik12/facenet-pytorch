import numpy as np
from feature_extract import ModelExtractFaceFeature
from PIL import Image
from glob import glob
import os

def data_load():
    model_eff = ModelExtractFaceFeature()
    img_dir = "./data/test_image4tani"
    img_save_flag = False

    feature_arr = np.empty((0,512), float)
    for img_path in glob(img_dir + '/*'):
        if img_save_flag:
            img_name = os.path.basename(img_path)
            img_save_dir = "./data/result/"
            img_save_path = img_save_dir + img_name
        else:
            img_save_path = None            

        img = Image.open(img_path)

        img_cropped = model_eff.trim_img(img.resize((160, 160)), model_eff.trim_face_model, img_path=img_save_path)
        #print(img_cropped.shape)

        feature = model_eff.inference(img_cropped, model_eff.extract_feature_model)
        feature_numpy = feature.to('cpu').detach().numpy().copy()
        feature_arr = np.append(feature_arr, np.array(feature_numpy), axis=0)
    
    #print(feature_arr.shape)
    return feature_arr, glob(img_dir + '/*')


if __name__ == "__main__":
    feat = data_load()

