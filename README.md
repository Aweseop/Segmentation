# Stratified K-Fold for Segmentation
This is a Pytorch implementation of Stratified K-Fold for Segmentation task. The dataset follows **COCO format**.

You can consider the following three conditions:

1. Number of categories for each image
2. Number of category types for each image
3. Number of annotations in each image

Categories can be grouped into lists and customized by referring to visualization and standard deviation for categories using **save_data.ipynb** as follows.

<br>

```
# type_list : Category types grouping
# num_list : Number of annotations grouping
# split_num : Fold number

type_list = [[1], [2, 3, 4], [5, 6], [7, 8]]
num_list = [23, 35]
split_num = 5
```

---

## Results
![init](https://user-images.githubusercontent.com/87693860/178210344-773bcff1-cee6-483d-a747-04878118c271.PNG)

<br>

![split_5Fold](https://user-images.githubusercontent.com/87693860/178210371-fa213b00-e4f0-4bb9-9bc3-e3f1d803dab1.PNG)
