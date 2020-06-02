#tani
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
import os

class ModelExtractFaceFeature:
    def __init__(self):
        self.trim_face_model, self.extract_feature_model = self.load_model()
        
    # モデルの読み込み
    def load_model(self):
        # 顔の切り取るためのモデル
        mtcnn = MTCNN(
            image_size=160, margin=0, min_face_size=20,
            thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
        )
        # 顔の特徴量抽出を抽出するためのモデル
        resnet = InceptionResnetV1(pretrained='vggface2').eval()
        
        return mtcnn, resnet

    # 顔領域をトリミングしたものを保存。
    def trim_img(self, img, model, img_path=None):
        if img_path==None:
            img_cropped = model(img)
        else:
            img_cropped = model(img, save_path=img_path)
        #print(img_cropped.shape)
        return img_cropped
    
    # 顔の特徴を抽出
    def inference(self, img_cropped, model):
        img_embedding = model(img_cropped.unsqueeze(0))
        #print(img_embedding.shape)
        return img_embedding

if __name__ == "__main__":
    img_path = "/Users/taniyan/git/facenet-pytorch/data/test_images/angelina_jolie/1.jpg"
    img = Image.open(img_path)

    dirname = os.path.dirname(img_path)
    img_save_path = None

    model_eff = ModelExtractFaceFeature()
    img_cropped = model_eff.trim_img(img.resize((160, 160)), model_eff.trim_face_model, img_path=img_save_path)
    feature = model_eff.inference(img_cropped, model_eff.extract_feature_model)
    feature_numpy = feature.to('cpu').detach().numpy().copy()