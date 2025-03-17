import numpy as np
from tqdm import tqdm
import pickle
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import cv2
from sklearn.linear_model import LogisticRegression

class Haar_feature():
    r"""
       haar feature 
    """
    def __init__(self, feature_type, position, size):
        self.feature_type = feature_type
        self.position = position
        self.size = size
        self.feature_values = []

    def compute_single_area(self, integral_image, x, y, width, height):
        """计算单个区域的特征值。"""
        s0 = integral_image[y             , x           ]
        s1 = integral_image[y             , x + width-1 ]
        s2 = integral_image[y + height-1  , x           ]
        s3 = integral_image[y + height-1  , x + width-1 ]
        return s3 + s0 - s1 - s2
            
    def compute(self, index, pre_computing=True,integral_images=None):
        if pre_computing :
            return np.array(self.feature_values)[index]
        x, y = self.position
        width, height = self.size
        feature_values = []
        if self.feature_type == 0 :
            """
            type A:
                [0,1]
                [0,1] 
            """
            feature_values = []           
            for integral_image in integral_images:

                left_feature    = self.compute_single_area(integral_image, x, y, width // 2, height)
                right_feature   = self.compute_single_area(integral_image, x + width // 2, y, width // 2, height)
                feature_value   = left_feature - right_feature            
                feature_values.append(feature_value)           

        elif self.feature_type == 1 :
            """
            type B:
                [1,1]
                [0,0] 
            """ 
            feature_values = []           
            for integral_image in integral_images:

                top_feature     = self.compute_single_area(integral_image, x, y, width, height // 2)
                bottom_feature  = self.compute_single_area(integral_image, x, y + height // 2, width, height // 2)
                feature_value   = bottom_feature - top_feature
                feature_values.append(feature_value)
                
        elif self.feature_type == 2 :
            """
            type C:
                [0,1,0]
                [0,1,0] 
            """
            feature_values = []           
            for integral_image in integral_images:

                left_feature    = self.compute_single_area(integral_image, x, y, width//3, height)
                middle_feature  = self.compute_single_area(integral_image, x+width//3, y, width//3, height)
                right_feature   = self.compute_single_area(integral_image, x+2*width//3, y ,width//3, height)
                feature_value   =   left_feature + right_feature - 2*middle_feature
                feature_values.append(feature_value)             

        else :
            """
            type D:
                [0,1]
                [1,0] 
            """
            feature_values = []           
            for integral_image in integral_images:

                left_top_feature    = self.compute_single_area(integral_image, x, y, width // 2, height // 2)
                right_top_feature   = self.compute_single_area(integral_image, x + width // 2, y, width // 2, height // 2)
                left_bottom_feature = self.compute_single_area(integral_image, x, y + height // 2, width // 2, height // 2)
                right_bottom_feature= self.compute_single_area(integral_image, x + width // 2, y + height // 2, width // 2, height // 2)
                feature_value = left_top_feature - right_top_feature + right_bottom_feature - left_bottom_feature
                feature_values.append(feature_value) 

        self.feature_values = np.array(feature_values)
        return self.feature_values
    
class Weak_haar_learners():
    r"""
        weak_learners -> for Adaboost 
    """
    def __init__(self, haar_feature:Haar_feature, threshold: float, polarity: int):
        r"""
            init:

            parameters:
            - haar_feature: used haar feature template
            - threshold: the threshold for classify
            - polarity: the direction for classify : 1/-1

        """
        self.haar_feature = haar_feature
        self.threshold = threshold
        self.polarity = polarity

    def predict(self, index, pre_computing=True, integral_images=None):
        r"""
            classify by the haar feature of image 

            input: 
            - index: dataset_index
            - pre_computing: weather to use the saved haar_features_values
            - integral_images: integral_images for computing haar_features

            return:
            - result: 1/-1 []
        """
        feature_values = self.haar_feature.compute(index, pre_computing=pre_computing,integral_images=integral_images) 
        predictions = np.ones(len(feature_values))     
        predictions[(self.polarity*feature_values) < (self.polarity * self.threshold)] = 0  
        return predictions

class Haar_Adaboost():

    r"""
        Adaboost algorithum with haar feature weak learner
    """
    def __init__(self, n_learners: int, haar_step: int, haar_type_num: int, image_size: list, feature_size: list, haar_features_path: str) -> None:
        r"""
            init:

            parameters:
            - n_learners: num of weak_learners
            - haar_step : the step of haar features
            - haar_type_num : template nums
            - image_size : (w,h)
            - feature_size : (min_w,min_h,max_w,max_h)
            - haar_features_path : '/data/haar_features.pkl'
        """
        self.n_learners = n_learners
        self.haar_step = haar_step
        self.haar_type_num = haar_type_num
        self.image_size = image_size
        self.feature_size = feature_size
        self.haar_features_path = haar_features_path
        self.weak_learners = []
        self.alphas = []

    def save_haar_features(self,haar_features, file_name):
        with open(file_name, 'wb') as file:
            pickle.dump(haar_features, file)
        
        print('Saving Haar-feature Successfully!')

    def load_haar_features(self,file_name):
        with open(file_name, 'rb') as file:
            haar_features = pickle.load(file)
        
        print('Loading Haar-feature Successfully!')
        return haar_features

    def compute_integral_images(self, images):
        r"""
            computing the integral_image of each image in the dataset 
        """

        integral_images = []
        for image in images:
            image = image.astype(np.float64)
            integral = np.cumsum(np.cumsum(image,axis=0),axis=1)
            integral_images.append(integral)

        return integral_images

    def compute_haar_feature(self, images):
        r"""
            computing the Haar_feature of each images
        """
        print('Without this Haar_features: Computing...')

        feature_params = self.generate_feature_params()    

        integral_images = self.compute_integral_images(images)

        haar_features = []
        for params in tqdm(feature_params):
            # 创建HaarFeature实例
            haar_feature = Haar_feature(params['feature_type'], params['position'], params['size'])
            # 计算当前特征类型的所有积分图的特征值
            feature_values = haar_feature.compute(index=0,pre_computing=False,integral_images=integral_images)
            # 保存计算结果
            haar_features.append(haar_feature)
        self.save_haar_features(haar_features,self.haar_features_path)
        
        print('Computing Finished !')

        return 0  

    def generate_feature_params(self):
        r"""
        generate params

        input:
        - image_width, image_height
        - min_feature_width, min_feature_height
        - max_feature_width, max_feature_height
        - step

        return:
        - feature_params
        """
        step = self.haar_step
        type_num = self.haar_type_num
        min_feature_width, min_feature_height, max_feature_width, max_feature_height = self.feature_size
        image_width, image_height = self.image_size
        feature_params = []
        for feature_type in range(type_num):  # 0 1 2 3 type
            for width in range(min_feature_width, max_feature_width + 1, step):
                for height in range(min_feature_height, max_feature_height + 1, step):
                    for x in range(0, image_width - width + 1, step):
                        for y in range(0, image_height - height + 1, step):
                            feature_params.append({
                                'feature_type': feature_type,
                                'position': (x, y),
                                'size': (width, height)
                            })
        return feature_params

    def find_best_threshold(self, feature_values, labels, weights):
        # sort by f_v
        sorted_idx = np.argsort(feature_values)
        sorted_feature_values = feature_values[sorted_idx]
        sorted_labels = labels[sorted_idx]
        sorted_weights = weights[sorted_idx]

        T1 = np.sum(sorted_weights[sorted_labels == 1]) # pos sum
        T0 = np.sum(sorted_weights[sorted_labels == 0]) # neg sum

        S1, S0 = 0, 0

        min_error = np.inf
        best_threshold = None
        best_polarity = None

        for i in range(1, len(sorted_feature_values)):
            # update S0 & S1:
            if sorted_labels[i-1] == 1:
                S1 += sorted_weights[i-1]
            else:
                S0 += sorted_weights[i-1]
            
            error1 = S1 + (T0 - S0)  # < threshold --> neg 
            error2 = S0 + (T1 - S1)  # < threshold --> pos
            error = min(error1, error2)
            
            if error < min_error:
                min_error = error
                best_threshold = (sorted_feature_values[i-1] + sorted_feature_values[i]) / 2
                best_polarity = -1 if error2 < error1 else 1

        return best_threshold, best_polarity, min_error
    
    def search_learner(self, haar_features,index, labels, weights, used_haar_features):

        r"""
            Given the integral_images to search the best weak_learner 
                in the (haar_feature & threshold & polarity)
            
            input:
            - haar_features
            - labels
            - weights

            output:
            - weak_learner
            - error
            - predictions
        """
        min_error = float('inf')
        best_learner = None
        best_predictions = None

        for feature in tqdm(haar_features):
            if feature in used_haar_features:
                continue
            
            feature_values = feature.compute(index,pre_computing=True)
            
            threshold, polarity, error = self.find_best_threshold(feature_values, labels, weights)
            
            if error < min_error:
                min_error = error
                best_learner = Weak_haar_learners(haar_feature=feature,threshold=threshold,polarity=polarity)
                
            best_predictions = best_learner.predict(index,pre_computing=True)    
        return best_learner, min_error, best_predictions
    
    def generate_windows_size(self,image, scale_factor):
        r"""
           generate the windows_size:

        """
        window_sizes = []
        image_shape = image.shape
        current_size = [int(image_shape[0]*0.02),int(image_shape[1]*0.02)]
        max_size = [int(image_shape[0]*0.5),int(image_shape[1]*0.5)]

        while current_size[0] <= max_size[0] and current_size[1] <= max_size[1]:
            # 确保窗口不大于图像尺寸
            if current_size[0] <= image_shape[0] and current_size[1] <= image_shape[1]:
                window_sizes.append(current_size)
            
            # 计算下一个窗口尺寸
            next_size = (int(current_size[0] * scale_factor), int(current_size[1] * scale_factor))
            
            # 防止无限循环
            if next_size == current_size:
                break
            
            current_size = next_size

        return window_sizes

    def slide_windows(self, image, step, windows_size):
        r"""
            generate the windows:

        """
        for window_size in windows_size:
            for y in range(0, image.shape[0] - window_size[1], step):
                for x in range(0, image.shape[1] - window_size[0], step):
                    yield (x, y, image[ y:y + window_size[1], x:x + window_size[0]], window_size[0], window_size[1])

    def iou_calculate(self,boxA,boxB):
        r"""
            calculating the iou of boxA & boxB
        """
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
        boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
        boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

        iou = interArea / float(boxAArea + boxBArea - interArea)
        return iou        
    
    def NMS(self,boxes, overlap_threshold):
        r"""
            select the final box from the boxes
        """
        if len(boxes) == 0:
            return []

        if boxes.dtype.kind == "i":
            boxes = boxes.astype("float")

        pick = []
        x1 = boxes[:,0]
        y1 = boxes[:,1]
        x2 = boxes[:,2]
        y2 = boxes[:,3]

        area = (x2 - x1 + 1) * (y2 - y1 + 1)
        idxs = np.argsort(y2)

        while len(idxs) > 0:
            last = len(idxs) - 1
            i = idxs[last]
            pick.append(i)
            suppress = [last]

            for pos in range(0, last):
                j = idxs[pos]

                xx1 = max(x1[i], x1[j])
                yy1 = max(y1[i], y1[j])
                xx2 = min(x2[i], x2[j])
                yy2 = min(y2[i], y2[j])

                w = max(0, xx2 - xx1 + 1)
                h = max(0, yy2 - yy1 + 1)

                overlap = float(w * h) / area[j]

                if overlap > overlap_threshold:
                    suppress.append(pos)

            idxs = np.delete(idxs, suppress)

        return boxes[pick].astype("int")        
    
    def train(self, images, labels, index):
        r"""
            training 

            input:
            - images: training data
            - labels: y of image(1/0)
            - index  : train / valid/ test
        """

        images = images
        labels = labels
        images = np.stack(images,axis=0)
        labels = np.array(labels)
        train_index, valid_index, test_index = index
        positive_count = np.sum(np.where(labels[train_index] == 1, 1, 0))
        negative_count = np.sum(np.where(labels[train_index] == 0, 1, 0))
        weights = np.where(labels[train_index] == 1,
                           1/(2*positive_count),
                           1/(2*negative_count))

        T = self.n_learners

        haar_features = self.load_haar_features(self.haar_features_path)

        train_metrics = []
        valid_metrics = []
        test_metrics  = []

        used_haar_features = []
        for t in range(T):

            weak_learner, error, predictions = self.search_learner(haar_features, train_index, labels[train_index], weights, used_haar_features)
            used_haar_features.append(weak_learner.haar_feature)
            beta = error/(1-error)
            alpha = np.log(1/beta)            
            weights *= np.exp(-alpha * np.where(labels == predictions, 1, 0))
            weights /= np.sum(weights)


            print(f"""weak_learner:
            1.haar_feature : {
                weak_learner.haar_feature.feature_type,
                weak_learner.haar_feature.position,
                weak_learner.haar_feature.size
            }
            2.threshold : {weak_learner.threshold} 
            3.polarity  : {weak_learner.polarity}
            """)

            self.weak_learners.append(weak_learner)
            self.alphas.append(alpha)
        
            train_prediction =  self.predict(train_index,None,inference=False)
            valid_prediction =  self.predict(valid_index,None,inference=False)
            test_prediction  =  self.predict(test_index ,None,inference=False)

            train_accuracy = accuracy_score(labels[train_index], train_prediction)
            valid_accuracy = accuracy_score(labels[valid_index], valid_prediction)
            test_accuracy = accuracy_score(labels[test_index], test_prediction)
            train_auc = roc_auc_score(labels[train_index], train_prediction)
            valid_auc = roc_auc_score(labels[valid_index], valid_prediction)
            test_auc = roc_auc_score(labels[test_index], test_prediction)
            train_metrics.append([train_accuracy,train_auc])
            valid_metrics.append([valid_accuracy,valid_auc])
            test_metrics.append([test_accuracy,test_auc])

            print(f'epoch: {t+1} error: {error}  train_auc: {train_auc},    valid_auc: {valid_auc},    test_auc: {test_auc}')
        return [train_metrics, valid_metrics, test_metrics]

    def predict(self, index, images, inference=False):
        r"""
            predict the final classes of images with the weak_learners & images

            input:
            - images:

            output:
            - predictions: 
        """
        if inference:

            integral_images = self.compute_integral_images(images=images)

            predictions = np.zeros(len(integral_images))
            for i in range(len(self.weak_learners)):
                predictions += self.alphas[i]*self.weak_learners[i].predict(None,False,integral_images)
            alpha = 0.5*sum(self.alphas)
            predictions = [ 1 if prediction >= alpha else 0 for prediction in predictions]
            return predictions
        
        else:
            predictions = np.zeros(len(index))
            for i in range(len(self.weak_learners)):
                predictions += self.alphas[i]*self.weak_learners[i].predict(index,True)
            alpha = 0.5*sum(self.alphas)
            predictions = [ 1 if prediction >= alpha else 0 for prediction in predictions]
            return predictions

    def inference(self, images, true_bounding_box, factor):
        r"""
            inference a new image to get the bounding_box of faces included

            input:
            - images
            - bounding_box

            output:
            - forcast_bounding_box
            - IoU score of performance
        """
        stepSize = 10
        final_boxes = []
        IoUs = []

        for i,image in enumerate(images):
            print(f'Inferencing for {i+1} th image ...')
            boxes = []        
            windows_size = self.generate_windows_size(image,factor)
            for (x, y, window, winW, winH) in tqdm(self.slide_windows(image, step=stepSize, windows_size=windows_size)):

                window = cv2.resize(window, self.image_size)
                prediction = self.predict(index=None,images=[window],inference=True)

                if prediction[0] == 1:
                    boxes.append((x, y, x + winW, y + winH))

            boxes = np.array(boxes)
            if len(boxes) == 0:
                continue
            boxes = self.NMS(boxes, 0.2)

            final_boxes.append(boxes.tolist())
            
            IoUs.append([self.iou_calculate(pred_box, gt_box) for pred_box in boxes for gt_box in true_bounding_box[i][1:]])

        return final_boxes, IoUs    
    

class Logistic(Haar_Adaboost):
    def __init__(self, n_learners: int, haar_step: int, haar_type_num: int, image_size: list, feature_size: list, haar_features_path: str) -> None:
        super().__init__(n_learners, haar_step, haar_type_num, image_size, feature_size, haar_features_path)
        self.log = LogisticRegression()
        self.haar_features_path = haar_features_path


    def train(self, images, labels, index):
        r"""
            train the logistic by haar features
        """
        labels = np.array(labels)
        self.haar_features = self.load_haar_features(self.haar_features_path)
        haar_features_values = [haar_feature.feature_values for haar_feature in self.haar_features]
        self.haar_features_values = np.array(haar_features_values).T
        train_index, valid_index, test_index = index
        self.log.fit(self.haar_features_values[train_index], labels[train_index])

        train_metrics = []
        valid_metrics = []
        test_metrics  = []

        train_prediction =  self.predict(train_index,None,inference=False)
        valid_prediction =  self.predict(valid_index,None,inference=False)
        test_prediction  =  self.predict(test_index ,None,inference=False)

        train_accuracy = accuracy_score(labels[train_index], train_prediction)
        valid_accuracy = accuracy_score(labels[valid_index], valid_prediction)
        test_accuracy = accuracy_score(labels[test_index], test_prediction)
        train_auc = roc_auc_score(labels[train_index], train_prediction)
        valid_auc = roc_auc_score(labels[valid_index], valid_prediction)
        test_auc = roc_auc_score(labels[test_index], test_prediction)
        train_metrics.append([train_accuracy,train_auc])
        valid_metrics.append([valid_accuracy,valid_auc])
        test_metrics.append([test_accuracy,test_auc])

        print(f'train_auc: {train_auc},    valid_auc: {valid_auc},    test_auc: {test_auc}')

        return [train_metrics, valid_metrics, test_metrics]

    def predict(self, index, images, inference=False):
        r"""
            predict the final classes of images with the weak_learners & images

            input:
            - images:

            output:
            - predictions: 
        """
        if inference:

            integral_images = self.compute_integral_images(images=images)

            haar_features_values = [haar_feature.compute(index=None, pre_computing=False, integral_images=integral_images) for haar_feature in self.haar_features]
            predictions = self.log.predict(np.array(haar_features_values).T)
            return predictions
        
        else:

            haar_features_values = self.haar_features_values[index]
            predictions = self.log.predict(haar_features_values)
            return predictions
        