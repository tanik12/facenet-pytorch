#tani                                                                                                                    
from utility.feature_extract import ModelExtractFaceFeature
from utility.data_loader import data_load
from PIL import Image
import numpy as np
import os

#ノルムの計算時にブロードキャストが行われないので注意が必要
def cos_sim(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

def ranking_cos_sim(img_path, img_list, res):
    #Debug用.一番高い値を使いたい場合使用すること.
    res = np.sum(res, axis=0).reshape(1, -1)
    max_id = np.argmax(res, 1)
    img_path = img_list[max_id[0]]
    sim_val = res[0, max_id[0]]
    print(sim_val, img_path)

    rank_val = 5
    sorted_ids = res[0].argsort()[::-1]
    sim_val_top5 = res[0, sorted_ids][:rank_val]
    img_name_top5 = np.array(img_list)[sorted_ids][:rank_val]

    print("類似度が高い順に並び替えています。1に近ければ近いほど似ている。")
    for i, (val, name) in enumerate(zip(sim_val_top5, img_name_top5)):
        person_name = os.path.basename(name).replace(".jpg", "")
        sentence = "No.{0}: (類似度, 名前) --> ({1}, {2})".format(i, val, person_name)
        print(sentence)

def euclid_sim(v1, v2):
    res = np.array([np.linalg.norm(v1[i]-v2[i]) for i in range(v1.shape[0])])
    return res

def ranking_euclid_sim(img_path, img_list, res):
    #Debug用.一番高い値を使いたい場合使用すること.
    res = res.reshape(1, -1)
    min_id = np.argmin(res, 1)
    img_path = img_list[min_id[0]]
    sim_val = res[0, min_id[0]]
    print(sim_val, img_path)

    rank_val = 5
    sorted_ids = res[0].argsort()
    sim_val_top5 = res[0, sorted_ids][:rank_val]
    img_name_top5 = np.array(img_list)[sorted_ids][:rank_val]

    print("類似度が高い順に並び替えています。0に近ければ近いほど似ている。1.1以上は別人。")
    for i, (val, name) in enumerate(zip(sim_val_top5, img_name_top5)):
        person_name = os.path.basename(name).replace(".jpg", "")
        sentence = "No.{0}: (類似度, 名前) --> ({1}, {2})".format(i, val, person_name)
        print(sentence)

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

    feature_numpy = np.tile(feature_numpy, (feat_compare.shape[0], 1))

    res = cos_sim(feature_numpy, feat_compare.T)
    ranking_cos_sim(img_path, img_list, res)

    res_2 = euclid_sim(feature_numpy, feat_compare)
    ranking_euclid_sim(img_path, img_list, res_2)

if __name__ == "__main__":
    main()
