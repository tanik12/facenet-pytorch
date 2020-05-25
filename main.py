#tani                                                                                                                    
from demo_code import ModelExtractFaceFeature
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
import os

def main():
    img_path = "/Users/taniyan/git/facenet-pytorch/data/test_images/angelina_jolie/1.jpg"
    img = Image.open(img_path)
    dirname = os.path.dirname(img_path)
    model_eff = ModelExtractFaceFeature()

    img_cropped = model_eff.trim_img(img, dirname, model_eff.trim_face_model)
    feature = model_eff.inference(img_cropped, model_eff.extract_feature_model)
    
    print(feature.shape)

if __name__ == "__main__":
    main()
