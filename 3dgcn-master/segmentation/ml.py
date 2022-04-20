import os
import json
from sklearn.cluster import KMeans
from sklearn.svm import SVC
import torch
import numpy as np
import pandas as pd
from collections import Counter

PART_NUM = {
    "Airplane": 4,
    "Bag": 2,
    "Cap": 2,
    "Car": 4,
    "Chair": 4,
    "Earphone": 3,
    "Guitar": 3,
    "Knife": 2,
    "Lamp": 4,
    "Laptop": 2,
    "Motorbike": 6,
    "Mug": 2,
    "Pistol": 3,
    "Rocket": 3,
    "Skateboard": 3,
    "Table": 3,
}
def get_valid_labels(category: str):
    # print(category)
    assert category in PART_NUM
    base = 0
    for cat, num in PART_NUM.items():
        if category == cat:
            valid_labels = [base + i for i in range(num)]
            return valid_labels
        else:
            base += num
def rearrange(l1, l2):
    # dict1={}
    # dict2={}
    # for i in l1:
    #     if dict1.__contains__(str(i)):
    #         dict1[str(i)]+=1
    #     else:
    #         dict1[str(i)]=0
    # for j in l2:
    #     if dict2.__contains__(str(j)):
    #         dict2[str(j)]+=1
    #     else:
    #         dict2[str(j)]=0
    c1=Counter(l1).most_common()
    c2=Counter(l2).most_common()
    dict={}
    for i in range(len(c1)):
        dict[c1[i][0]]=c2[i][0]
    l3=[]
    for i in l1:
        l3.append(dict[i])
    return l3
def get_miou(pred: "tensor (point_num, )", target: "tensor (point_num, )", valid_labels: list):
    pred, target = pred.cpu().numpy(), target.cpu().numpy()
    part_ious = []
    for part_id in valid_labels:
        pred_part = (pred == part_id)
        target_part = (target == part_id)
        I = np.sum(np.logical_and(pred_part, target_part))
        U = np.sum(np.logical_or( pred_part, target_part))
        if U == 0:
            part_ious.append(1)
        else:
            part_ious.append(I/U)
    miou = np.mean(part_ious)
    return miou

class IouTable():
    def __init__(self):
        self.obj_miou = {}

    def add_obj_miou(self, category: str, miou: float):
        if category not in self.obj_miou:
            self.obj_miou[category] = [miou]
        else:
            self.obj_miou[category].append(miou)

    def get_category_miou(self):
        """
        Return: moiu table of each category
        """
        category_miou = {}
        for c, mious in self.obj_miou.items():
            category_miou[c] = np.mean(mious)
        return category_miou

    def get_mean_category_miou(self):
        category_miou = []
        for c, mious in self.obj_miou.items():
            c_miou = np.mean(mious)
            category_miou.append(c_miou)
        return np.mean(category_miou)

    def get_mean_instance_miou(self):
        object_miou = []
        for c, mious in self.obj_miou.items():
            object_miou += mious
        return np.mean(object_miou)

    def get_string(self):
        mean_c_miou = self.get_mean_category_miou()
        mean_i_miou = self.get_mean_instance_miou()
        first_row = "| {:5} | {:5} ||".format("Avg_c", "Avg_i")
        second_row = "| {:.3f} | {:.3f} ||".format(mean_c_miou, mean_i_miou)

        categories = list(self.obj_miou.keys())
        categories.sort()
        cate_miou = self.get_category_miou()

        for c in categories:
            miou = cate_miou[c]
            first_row += " {:5} |".format(c[:3])
            second_row += " {:.3f} |".format(miou)

        string = first_row + "\n" + second_row
        return string

class snds():
    def __init__(self,
                 root: str,
                 split: str = 'test',
                 transform=None):
        super().__init__()
        self.root = root
        self.transform = transform


        # Set category
        self.category_id = {}
        print(root)
        with open(os.path.join(root, "synsetoffset2category.txt")) as cat_file:
            for line in cat_file:
                tokens = line.strip().split()
                self.category_id[tokens[1]] = tokens[0]
        self.category_names = list(self.category_id.values())

        # Read split file
        split_file_path = os.path.join(root, "train_test_split", "shuffled_{}_file_list.json".format(split))
        split_file_list = json.load(open(split_file_path, "r"))
        cat_ids = list(self.category_id.keys())
        self.file_list = []
        for name in split_file_list:
            _, cat_id, obj_id = name.strip().split("/")
            if cat_id in cat_ids:
                self.file_list.append(os.path.join(cat_id, obj_id))
        self.file_num=len(self.file_list)
    def readfile(self,index):
        # change one
        # origin:cat_id, obj_id = self.file_list[index].split("/")
        cat_id, obj_id = self.file_list[index].split("\\")
        category = self.category_id[cat_id]

        points = torch.FloatTensor(
            np.genfromtxt(os.path.join(self.root, cat_id, "points", "{}.pts".format(obj_id))))
        labels = torch.LongTensor(
            np.genfromtxt(os.path.join(self.root, cat_id, "points_label", "{}.seg".format(obj_id))))
        labels = labels - 1 + get_valid_labels(category)[0]
        # sample_ids = torch.multinomial(torch.ones(points.size(0)), num_samples=self.point_num, replacement=True)
        # points = points[sample_ids]
        # labels = labels[sample_ids]
        if self.transform:
            points = self.transform(points)
        return category, obj_id, points, labels
    def calculate_save_mious(self, iou_table, category_names, labels, predictions):
        for i in range(len(category_names)):
            category = category_names[i]
            pred = predictions[i]
            label =  labels[i]
            valid_labels = get_valid_labels(category)
            miou = get_miou(pred, label, valid_labels)
            iou_table.add_obj_miou(category, miou)


    def kmean(self):
        ioudict={}
        train_iou_table = IouTable()
        for i in range(self.file_num):
            cat,oid,p,label=self.readfile(i)
            l=label.tolist()
            c1 = Counter(l).most_common()
            classnum=len(c1)
            classifier=KMeans(n_clusters=classnum)
            classifier.fit(p)
            result=classifier.labels_
            result=rearrange(result,l)
            result=torch.tensor(result)

            self.calculate_save_mious(train_iou_table, [cat], label, result)
            c_miou = train_iou_table.get_mean_category_miou()
            i_miou = train_iou_table.get_mean_instance_miou()
            print('Progress({:.3f})miou(c): {:.3f} | miou(i): {:.3f}'.format(i/self.file_num, c_miou, i_miou))
        train_table_str = train_iou_table.get_string()
        print(train_table_str)
        #     if ioudict.__contains__(cat):
        #         ioudict[cat].append(c_miou)
        #     else:
        #         ioudict[cat]=[c_miou]
        # for key in ioudict:
        #     ave=sum(ioudict[key])/len(ioudict[key])
        #     print('Average mIOU of class %s is :%f'%(key,ave))

    def trainsvm(self,modeldict):
        ioudict = {}
        datadict={}
        labeldict={}
        train_iou_table = IouTable()
        for i in range(self.file_num):
            cat, oid, p, label = self.readfile(i)
            b=len(datadict)
            if len(datadict)!=16:
                if cat in datadict:
                    c=datadict[cat].shape[0]
                    if datadict[cat].shape[0]<=2000:
                        datadict[cat]=torch.cat([datadict[cat],p],dim=0)

                        labeldict[cat]=torch.cat([labeldict[cat],label],dim=0)
                else:
                    datadict[cat]=p
                    labeldict[cat]=label
                print('Progress({:.3f})'.format(i / self.file_num))
        n=0
        for cat in PART_NUM:
            if 1:
                modeldict[cat]=SVC(C=1.0, kernel='rbf', gamma='auto', decision_function_shape='ovr', cache_size=500)
                modeldict[cat].fit(datadict[cat],labeldict[cat])
                # print(n)
                # n+=1
    def testsvm(self,modeldict):
        train_iou_table = IouTable()
        for i in range(self.file_num):
            cat,oid,p,label=self.readfile(i)
            if 1:

                classifier=modeldict[cat]
                result=classifier.predict(p)
                result=torch.tensor(result)

                self.calculate_save_mious(train_iou_table, [cat], label, result)
                c_miou = train_iou_table.get_mean_category_miou()
                i_miou = train_iou_table.get_mean_instance_miou()
                print('Progress({:.3f})miou(c): {:.3f} | miou(i): {:.3f}'.format(i/self.file_num, c_miou, i_miou))
        train_table_str = train_iou_table.get_string()
        print(train_table_str)

    # def keam(self):
    #     l=label.tolist()
    #     c1 = Counter(l).most_common()
    #     classnum=len(c1)
    #     c1 = Counter(l).most_common()
    #     classnum = len(c1)
    #     classifier = KMeans(n_clusters=classnum)
    #     classifier.fit(p)
    #     result = classifier.labels_
    #     result = rearrange(result, l)
    #     result = torch.tensor(result)
    #
    #     self.calculate_save_mious(train_iou_table, [cat], label, result)
    #     c_miou = train_iou_table.get_mean_category_miou()
    #     i_miou = train_iou_table.get_mean_instance_miou()
    #     print('Progress({:.3f})miou(c): {:.3f} | miou(i): {:.3f}'.format(i / self.file_num, c_miou, i_miou))
    #     train_table_str = train_iou_table.get_string()
    #     print(train_table_str)













            #     c1 = Counter(l).most_common()
            #     classnum = len(c1)
            #     classifier = KMeans(n_clusters=classnum)
            #     classifier.fit(p)
            #     result = classifier.labels_
            #     result = rearrange(result, l)
            #     result = torch.tensor(result)
            #
            #     self.calculate_save_mious(train_iou_table, [cat], label, result)
            #     c_miou = train_iou_table.get_mean_category_miou()
            #     i_miou = train_iou_table.get_mean_instance_miou()
            #     print('Progress({:.3f})miou(c): {:.3f} | miou(i): {:.3f}'.format(i / self.file_num, c_miou, i_miou))
            # train_table_str = train_iou_table.get_string()
            # print(train_table_str)

c=snds(root="../../shapenetcore_partanno_segmentation_benchmark_v0")
c.train()

# modeldict={}
# train_data = snds(root="../../shapenetcore_partanno_segmentation_benchmark_v0", split= 'train')
# train_data.trainsvm(modeldict)
# test_data=snds(root="../../shapenetcore_partanno_segmentation_benchmark_v0", split= 'test')
# test_data.testsvm(modeldict)

# rearrange()
# print(1)