#tani                                                                                                                    
from feature_extract import ModelExtractFaceFeature
from PIL import Image
import os

def main():
    img_path = "/Users/taniyan/git/facenet-pytorch/data/test_images/angelina_jolie/1.jpg"
    img_save_path = None
    
    img = Image.open(img_path)
    dirname = os.path.dirname(img_path)
    model_eff = ModelExtractFaceFeature()

    img_cropped = model_eff.trim_img(img.resize((160, 160)), model_eff.trim_face_model, img_path=img_save_path)
    feature = model_eff.inference(img_cropped, model_eff.extract_feature_model)
    
    print(feature.shape)

if __name__ == "__main__":
    main()
