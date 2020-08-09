#tani                                                                                                                    
from utility.feature_extract import ModelExtractFaceFeature
from utility.data_loader import data_load
from utility.similarity_calculate import *

from PIL import Image
import numpy as np
import os

def main():
    feat_compare, img_list = data_load()

    img_path = "/Users/taniyan/git/facenet-pytorch/data/test_images/angelina_jolie/ayaueto.jpg"
    img_save_path = None
    
    img = Image.open(img_path)
    dirname = os.path.dirname(img_path)
    model_eff = ModelExtractFaceFeature()

    img_cropped = model_eff.trim_img(img.resize((160, 160)), model_eff.trim_face_model, img_path=img_save_path)
    feature = model_eff.inference(img_cropped, model_eff.extract_feature_model)
    feature_numpy = feature.to('cpu').detach().numpy().copy()

    feature_numpy = np.tile(feature_numpy, (feat_compare[0].shape[0], 1))
    # print("---> ", feature_numpy.shape, feat_compare[0].shape)

    #cos類似度
    res = cos_sim(feature_numpy, feat_compare[0].T)
    ranking_cos_sim(img_path, img_list, res)

    #ユークリッド距離(論文ではこちらを採用する)
    res_2 = euclid_sim(feature_numpy, feat_compare[0])
    ranking_euclid_sim(img_path, img_list, res_2)

if __name__ == "__main__":
    main()
