#!pip install iterative-stratification
import json
import copy
import numpy as np
import pandas as pd
from pycocotools.coco import COCO
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold



# Set the initial parameter
dataset_path = '/opt/ml/segmentation/input/data'
anns_file_path = dataset_path + '/' + 'train_all.json'
coco = COCO(anns_file_path)
with open(anns_file_path, "r") as f:
    dataset = json.loads(f.read())

categories = dataset['categories']
cat_names = [category['name'] for category in categories]

# Number of annotations in each category
def category_annotations(data) :
    cat_dist = np.zeros(len(data['categories']), dtype=int)
    
    for ann in data['annotations']:
        cat_dist[ann['category_id']-1] += 1
    
    return cat_dist.reshape(len(cat_dist), 1)


# Get the dataframe to create a label.
def get_dataframe(data) :
    img_ids = [ann['image_id'] for ann in data['annotations']]
    cat_ids = [ann['category_id'] for ann in data['annotations']]
    df = pd.DataFrame({'image_id' : img_ids, 'category_id' : cat_ids})
    df = df.sort_values(by=['image_id'], ascending=True).reset_index(drop=True)
    
    return df


# Get the categories in each image.
def get_cat_ann(data) :
    '''
    The fewest anns categories are to be included in each fold.
    Accordingly, a category having the smallest number of anns per image is designated as a category of the corresponding image.
    '''
    cat_ann = []
    df = get_dataframe(data)
    cat_dist = category_annotations(data)
    imgs = list(set(df['image_id']))
    
    for img_id in imgs:
        ann_ids = coco.getAnnIds(img_id)
        ann_lst = coco.loadAnns(ann_ids)
        ann_num = int(1e9)
        for ann in ann_lst:
            category = ann['category_id']
            if cat_dist[category-1] < ann_num:
                ann_num = cat_dist[category-1]
                img_class = category-1
        cat_ann.append(img_class)
        
    return cat_ann


# Parsing function for get_cat_type
def parse_values(x, divide_list) :
    num = len(set(x))

    for i, element in enumerate(divide_list) :
        if num in element :
            return i


# Get the number of category types in each image.
def get_cat_type(df, type_list) :
    all_element = []
    all_cat_num = df.groupby('image_id')['category_id'].apply(lambda x: len(set(x))).values
    
    for list_ele in type_list :
        all_element += list_ele
        
    if len(type_list) <= 1 :
        raise ValueError('Input must have at least two list')        
        
    if  len(all_element) != len(list(set(all_element))) :
        raise ValueError('Should be no duplicate value entered.')
        
    cat_type = df.groupby('image_id')['category_id'].apply(lambda x: parse_values(x, type_list)).values
    
    return cat_type


# Get the number of annotations in each image.
def get_ann_num(df, section_list) :
    ann_num = []
    section_list.sort()
    all_ann_num = df.groupby('image_id').apply(lambda x: len(x)).values
    
    for num in section_list :
        if num not in all_ann_num :
            raise ValueError('Out of range or Not integer list') 
    
    for num in all_ann_num :
        for i, input_num in enumerate(section_list):
            if num <= input_num :
                ann_num.append(i)
                break
            elif num > max(section_list) :
                ann_num.append(len(section_list)+1)
                break
                
    return ann_num


# Get the multi-label
def get_label(data, type_list, num_list) :
    df = get_dataframe(data)
    cat_ann = get_cat_ann(data)
    cat_type = get_cat_type(df, type_list)
    ann_num = get_ann_num(df, num_list)
    
    return cat_ann, cat_type, ann_num


# Get the X, y for stf k-fold
def get_id(data, type_list, num_list) :
    df = get_dataframe(data)
    img_ids = list(set(df['image_id']))
    cat_ann, cat_type, ann_num = get_label(data, type_list, num_list)
    
    X = np.array([data['images'][img_id]['id'] for img_id in img_ids])
    y = np.array([[cla, cat, ann] for cla, cat, ann in zip(cat_ann, cat_type, ann_num)])
    
    return X, y


# Check the number of categories in each fold
def check_categories(num, data) :
    cat_check = []
    data_cat_list = list(set([cat_names[ann['category_id'] - 1] for ann in data['annotations']]))
    
    for i, name in enumerate(cat_names) :
        if name in data_cat_list :
            cat_check.append(i)
            
    if len(cat_check) == len(cat_names) :
        print(f'Fold {num} has all categories')
    else :
        print(f'Fold {num} has just {len(cat_check)} categories')


# Split data using MultilabelStratifiedKFold
def split_data(mode, data, split_num, type_list, num_list) :
    fold_num = 1
    
    train_json_list =[]
    train_data_list =[]
    val_json_list =[]
    val_data_list =[]
    
    mskf = MultilabelStratifiedKFold(n_splits=split_num, shuffle=True, random_state=42)
    X, y = get_id(data, type_list, num_list)
    
    for train_index, val_index in mskf.split(X, y):
        train_json = f'kfold_{fold_num}_train.json'
        val_json = f'kfold_{fold_num}_val.json'

        X_train, X_val = X[train_index], X[val_index]

        train_images_id = coco.getImgIds(X_train)
        train_images = coco.loadImgs(train_images_id)
        
        train_anns_id = coco.getAnnIds(train_images_id)
        train_anns = coco.loadAnns(train_anns_id)
        
        val_images_id = coco.getImgIds(X_val)
        val_images = coco.loadImgs(val_images_id)
        
        val_anns_id = coco.getAnnIds(val_images_id)
        val_anns = coco.loadAnns(val_anns_id)
        
        train_data = {
            "info": data['info'],
            "licenses": data['licenses'],
            "images": train_images,
            "categories":data['categories'],
            "annotations": train_anns,
        }
        val_data = {
            "info": data['info'],
            "licenses": data['licenses'],
            "images": val_images,
            "categories":data['categories'],
            "annotations": val_anns,
        }
        
        train_json_list.append(train_json)
        train_data_list.append(train_data)
        val_json_list.append(val_json)
        val_data_list.append(val_data)
        
        fold_num += 1
        
    # Show our metrics that standard deviation in each label.
    if mode == 'metric' :
        cat_ann_list = []
        cat_type_list = []
        ann_num_list = []
        
        for val_data in val_data_list :
            '''
            Dividing the maximum of annotations in each category for scale in cat_ann_std.
            '''
            val_df = get_dataframe(val_data)
            cat_ann = category_annotations(val_data)
            cat_type = val_df.groupby('image_id')['category_id'].apply(lambda x: len(set(x))).values
            cat_type = cat_type / max(cat_type)
            ann_num = val_df.groupby('image_id').apply(lambda x: len(x)).values
            ann_num = ann_num / max(ann_num)

            cat_ann_list.append(cat_ann)
            cat_type_list.append(cat_type)
            ann_num_list.append(ann_num)
        
        cat_ann_list = np.array(cat_ann_list)
        cat_ann_std = np.mean(np.std([cat_ann_list[:, i] / max(cat_ann_list[:, i]) for i in range(len(cat_ann_list[0]))], axis=0))
        cat_type_std = np.mean([np.std(element) for element in cat_type_list])
        ann_num_std = np.mean([np.std(element) for element in ann_num_list])

        print('cat_ann_std :', cat_ann_std)
        print('cat_type_std :', cat_type_std)
        print('ann_num_std :', ann_num_std)
    
    # Load split data
    elif mode == 'load' :
        for i, val_data in enumerate(val_data_list) :
            check_categories(i+1, val_data)
        
        return train_json_list, val_json_list, train_data_list, val_data_list

    else :
        raise ValueError('Please enter input such as \'metric\' or \'load\'')
    

# Sort json file
def sort_data(data_list) :
    sorted_list = []
    
    for index in range(len(data_list)) :
        cp = copy.deepcopy(data_list[index]) # Use for shared memory issue
        for i in range(len(cp['images'])) :
            origin_id = cp['images'][i]['id']
            cp['images'][i]['id'] = i
            for j in range(len(cp['annotations'])) :
                cp['annotations'][j]['id'] = j
                if cp['annotations'][j]['image_id'] == origin_id :
                    cp['annotations'][j]['image_id'] = i

        sorted_list.append(cp)

    return sorted_list


# Save json file
def save_data(train_json_list, val_json_list, train_data_list, val_data_list) :
    for i in range(len(train_json_list)) :
        with open(dataset_path + '/' + train_json_list[i], 'w') as train_writer:
            json.dump(train_data_list[i], train_writer)
        print(f'{train_json_list[i]} saved')

        with open(dataset_path + '/' + val_json_list[i], 'w') as val_writer:
            json.dump(val_data_list[i], val_writer)
        print(f'{val_json_list[i]} saved')
    
    print('Done')
