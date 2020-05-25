#tani                                                                                                                    
from feature_extract import ModelExtractFaceFeature
from data_loader import data_load
from PIL import Image
import numpy as np
import os

def cos_sim(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

def main():
    feat_compare, img_list = data_load()

    img_path = "/Users/taniyan/git/facenet-pytorch/data/test_images/angelina_jolie/1.jpg"
    img_save_path = None
    
    img = Image.open(img_path)
    dirname = os.path.dirname(img_path)
    model_eff = ModelExtractFaceFeature()

    img_cropped = model_eff.trim_img(img.resize((160, 160)), model_eff.trim_face_model, img_path=img_save_path)
    feature = model_eff.inference(img_cropped, model_eff.extract_feature_model)
    feature_numpy = feature.to('cpu').detach().numpy().copy()

    res = cos_sim(feature_numpy, feat_compare.T)
    #Debug用.一番高い値を使いたい場合使用すること.
    #max_id = np.argmax(res, 1)
    #img_path = img_list[max_id[0]]
    #sim_val = res[0, max_id[0]]
    #print(sim_val, img_path)

    rank_val = 5
    sorted_ids = res[0].argsort()[::-1]
    sim_val_top5 = res[0, sorted_ids][:rank_val]
    img_name_top5 = np.array(img_list)[sorted_ids][:rank_val]

    print("類似度が高い順に並び替えています。")
    for i, (val, name) in enumerate(zip(sim_val_top5, img_name_top5)):
        person_name = os.path.basename(name).replace(".jpg", "")
        sentence = "No.{0}: (類似度, 名前) --> ({1}, {2})".format(i, val, person_name)
        print(sentence)

if __name__ == "__main__":
    main()
