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

    rank_val = 10
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

    rank_val = 10
    sorted_ids = res[0].argsort()
    sim_val_top5 = res[0, sorted_ids][:rank_val]
    img_name_top5 = np.array(img_list)[sorted_ids][:rank_val]

    print("類似度が高い順に並び替えています。0に近ければ近いほど似ている。1.1以上は別人。")
    for i, (val, name) in enumerate(zip(sim_val_top5, img_name_top5)):
        person_name = os.path.basename(name).replace(".jpg", "")
        sentence = "No.{0}: (類似度, 名前) --> ({1}, {2})".format(i, val, person_name)
        print(sentence)


if __name__ == "__main__":
    pass
