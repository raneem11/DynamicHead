import torch
from torch.utils.data import Dataset
import cv2
import numpy as np
from detectron2.structures import BoxMode
from detectron2.data import detection_utils as utils


class WebDataset(Dataset):
    def __init__(self,
                 images_dict,
                 transform=None,
                 ):

        self.transform = transform
        self.images_dict = images_dict

    def __len__(self):
        return len(self.images_dict)

    def __getitem__(self, image_index):
        self.item_dict = self.images_dict[image_index]
        anns = self.item_dict['annotations']
        img = self.load_image()
        image_shape = img.shape[:2]
        bboxes = self.load_annotations(anns)
        sample = {'image':img, 'annot':bboxes}
        if self.transform:
            sample = self.transform(sample)
        for idx, annot in enumerate(anns):
            box = sample['annot'][idx].tolist()
            box = BoxMode.convert(box, annot["bbox_mode"], BoxMode.XYXY_ABS)
            annot['bbox'] = torch.tensor(box)[None, :][0]
            annot["bbox_mode"] = BoxMode.XYXY_ABS
        
        instances = utils.annotations_to_instances(
            anns, image_shape
        )
        target  = {'image': sample['image']}
        target["instances"] = utils.filter_empty_instances(instances)
        target["file_name"] = self.item_dict['file_name']

        return target

    def load_image(self):
        image_path = self.item_dict['file_name']
        img = cv2.imread(image_path)
        img = cv2.resize(img, (64,64))
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        return img.astype(np.float32)/255.0

    def load_annotations(self, anns):
        annotations = [item['bbox'] for item in anns]
        annotations = np.array(annotations)
        return annotations