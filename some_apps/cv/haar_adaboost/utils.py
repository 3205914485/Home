import cv2
import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
def random_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)

def preprocessing(image):
    r"""
        Resize & 2Gray
    """

    processed_img = cv2.resize(image, (48, 48))

    gray = cv2.cvtColor(processed_img, cv2.COLOR_BGR2GRAY)

        
    return gray

def load_data(paths):

    print('Loading data ...') 

    images_path, labels_path, train_index_path, valid_index_path, test_index_path, inferencing_images_path, ground_truth_bounding_boxes_path = paths

    image_paths = sorted(os.listdir(images_path), key=lambda x: int(x.split('.')[0]))
    pics = []
    for image_path in image_paths:
        full_image_path = os.path.join(images_path, image_path)
        pic = cv2.imread(full_image_path)
        preprocessed = preprocessing(pic)
        pics.append(preprocessed)

    inferencing_image_paths = sorted(os.listdir(inferencing_images_path), key=lambda x: int(x.split('.')[0]))
    inferencing_images = []
    for inferencing_image_path in inferencing_image_paths:
        full_image_path = os.path.join(inferencing_images_path, inferencing_image_path)
        pic = cv2.imread(full_image_path)
        pic = cv2.cvtColor(pic, cv2.COLOR_BGR2GRAY)
        inferencing_images.append(pic)

    labels = pd.read_csv(labels_path)['label'].tolist()
    train_index = pd.read_csv(train_index_path)['train_index'].tolist()
    valid_index = pd.read_csv(valid_index_path)['valid_index'].tolist()
    test_index  = pd.read_csv(test_index_path)['test_index'].tolist()
    ground_truth_bounding_boxes = np.load(ground_truth_bounding_boxes_path,allow_pickle=True)

    print('Loading successfully !')
    return pics, labels, train_index, valid_index, test_index, inferencing_images, ground_truth_bounding_boxes

class Visuliaztion():
    r"""
        plot metrics curve & ious score & bouding_boxex result
    """
    def __init__(self,model_name, metrics, images, final_boxes, IoUs, gt_bbx,save_path): 
        self.plot_metrics(model_name, metrics)
        for i in range(len(images)):
            predictions = final_boxes[i]  
            ground_truths = gt_bbx[i][1:]  
            save_path = f'annotated_image_{i}.jpg'  
            self.draw_boxes_and_save(images[i], predictions, ground_truths, save_path)
            print(f'Final best IoU: {max(IoUs[i])} for the {i}th pics')


    def plot_metrics(model_name, metrics):
        r"""
        draw metrics curve
        
        paras:
        - metrics: [train_metrics, valid_metrics, test_metrics]
        """
        train_metrics, valid_metrics, test_metrics = metrics
        
        epochs = range(1, len(train_metrics) + 1)
        
        # ACC
        plt.figure(figsize=(14, 6))
        plt.subplot(1, 2, 1)
        plt.plot(epochs, [m[0] for m in train_metrics], label='Train Accuracy')
        plt.plot(epochs, [m[0] for m in valid_metrics], label='Validation Accuracy')
        plt.plot(epochs, [m[0] for m in test_metrics], label='Test Accuracy')
        plt.title('Accuracy over epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.tight_layout()
        if not os.path.exists('result'):
            os.makedirs('result')
        plt.savefig(f'result/{model_name}_acc_curve.png')  # save
        plt.close()  

        # AUC
        plt.subplot(1, 2, 2)
        plt.plot(epochs, [m[1] for m in train_metrics], label='Train AUC')
        plt.plot(epochs, [m[1] for m in valid_metrics], label='Validation AUC')
        plt.plot(epochs, [m[1] for m in test_metrics], label='Test AUC')
        plt.title('AUC over epochs')
        plt.xlabel('Epoch')
        plt.ylabel('AUC')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'result/{model_name}_auc_curve.png')  # save
        plt.close()          

    def draw_boxes_and_save(image, predictions, ground_truths, save_path):
        r"""
        draw bbx

        parameters:
        - image:
        - predictions: [(x1, y1, x2, y2), ...]。
        - ground_truths: [(x1, y1, x2, y2), ...]。
        - save_path: 
        """
 
        for (x1, y1, x2, y2) in predictions:
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)


        for (x1, y1, x2, y2) in ground_truths:
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

        cv2.imwrite(save_path, image)
