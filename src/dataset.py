import numpy as np
import random
import cv2

def imgs_annot_aggregator(iter,train_im_list,data):
    print("Running imgs_annot_aggregator...")
    final = np.zeros((iter,4))
    bounding_boxes = []
    image_names = []
    instances_img = []
    amount_matches = []
    class_type = []
    for j in range(iter): 
        instances_img = []                        
        img = random.sample(train_im_list,1)
        image_names.append(img[0])
        for i in range(len(data['categories'])):
            if [data['categories'][i]['image_fname']] == img:
                instances_img.append(data['categories'][i]['id'])
        for l in range(len(instances_img)):
            for i in range(len(data['categories'])):
                if data['categories'][i]['id'] == instances_img[l]:
                    bounding_boxes.append(data['categories'][i]['bbox'])
                    class_type.append(data['categories'][i]['role'])
        amount_matches.append(len(instances_img))
    final = amount_matches, image_names, bounding_boxes, class_type
    return final, image_names

def bbox_points(data_annot,train_imgs):
    print("Running bbox_points...")
    name = []
    x_org = []
    y_org = []
    x_dist = []
    y_dist = []
    bbox = []
    class_type = []
    file_path = []
    itr = 0
    for i in range(len(data_annot[0])):
        for j in range(data_annot[0][i]):
            name.append(data_annot[1][i])
            class_type.append(data_annot[3][i])
            x_org.append(data_annot[2][j+itr][0])
            y_org.append(data_annot[2][j+itr][1])
            x_dist.append(data_annot[2][j+itr][2])
            y_dist.append(data_annot[2][j+itr][3])
            bbox.append([data_annot[2][j+itr][0],data_annot[2][j+itr][1],data_annot[2][j+itr][0]+data_annot[2][j+itr][2],data_annot[2][j+itr][1]+data_annot[2][j+itr][3]])
            file_path.append(os.path.join(train_imgs, data_annot[1][i]))
        itr = itr + data_annot[0][i]
    df = pd.DataFrame(
    {'name': name,
        'class': class_type,
        'x_org': x_org,
        'y_org': y_org,
        'x_dist': x_dist,
        'y_dist': y_dist,
        'bbox': bbox,
        'file_path': file_path
    })
    return df

def transformsXY(path, bb, transforms,new_size,ratio):
    x = cv2.imread(str(path)).astype(np.float32)
    x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)/255
    x = cv2.resize(x,(new_size,new_size) ) 
    bb[0] = int(bb[0]/ratio)
    bb[1] = int(bb[1]/ratio)
    bb[2] = int(bb[2]/ratio)
    bb[3] = int(bb[3]/ratio)
    return x, bb 


class AircraftDataset(Dataset):
    def __init__(self, paths, bb, y, transforms=False):
        self.paths = paths.values
        self.bb = bb.values
        self.y = y.values
    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self, idx):
        path = self.paths[idx]
        y_class = self.y[idx]
        y_bb = self.bb[idx]
        x = self.paths
        x, y_bb = transformsXY(str(path), np.array(self.bb[idx],dtype=np.int32), True,new_size,ratio)
        x.transpose(1, 0, 2).strides        
        return x, y_class, y_bb