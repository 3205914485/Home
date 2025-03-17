from models import Haar_Adaboost, Logistic
from utils import load_data, Visuliaztion
import os
import json

def main():
    
    model_name = 'Haar_Adaboost'

    n_learners = 1
    haar_step = 6
    haar_type_num = 4
    image_size = [48,48]
    feature_size = [8,8,24,24]
    haar_features_path = f'data/haar_features_{haar_step}.pkl'
    factor = 1.75

    images_path = 'data/pics'
    labels_path = 'data/labels.csv'
    train_index_path = 'data/train_index.csv'
    valid_index_path = 'data/valid_index.csv'   
    test_index_path  = 'data/test_index.csv'   
    inferencing_images_path = 'data/infer_pics' 
    ground_truth_bounding_boxes_path = 'data/g_t_bounding_boxes.npy'
    save_path = 'data/bbx_pics'

    paths = [images_path, labels_path, train_index_path, \
            valid_index_path, test_index_path, inferencing_images_path, ground_truth_bounding_boxes_path]

    images, labels, train_index, valid_index, test_index , \
        inferencing_images, ground_truth_bounding_boxes= load_data(paths)
    
    if model_name == 'Haar_Adaboost':
        model = Haar_Adaboost(n_learners=n_learners,
                            haar_step=haar_step,
                            haar_type_num=haar_type_num,
                            image_size=image_size,
                            feature_size=feature_size,
                            haar_features_path=haar_features_path
                            )
    elif model_name == 'Logistic':
        model = Logistic(n_learners=n_learners,
                            haar_step=haar_step,
                            haar_type_num=haar_type_num,
                            image_size=image_size,
                            feature_size=feature_size,
                            haar_features_path=haar_features_path
                            )        
    else :
        raise ValueError(f"     Unsupported Model :     {model_name}    ")

    if not os.path.exists(haar_features_path):
        model.compute_haar_feature(images)
    
    metrics = model.train(images=images,
                labels=labels,
                index=[train_index,valid_index,test_index]
                )

    final_boxes, IoUs = model.inference(images=inferencing_images,
                    true_bounding_box=ground_truth_bounding_boxes,
                    factor=factor
                )

    Visuliaztion(model_name,metrics,inferencing_images,final_boxes,IoUs,ground_truth_bounding_boxes,save_path)

if __name__ == "__main__":
    main()