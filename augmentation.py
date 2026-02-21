# import albumentations as A
# #Convert image to torch.Tensor and divide by 255 if image or mask are uint8 type.
# # from albumentations.pytorch import ToTensor
# import cv2
# import numpy as np
#  # mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
# def get_augmentation_gray(image_size, train_flag=True):
#     #image_size tuple or list of [height, width]
#     small_len = min(image_size)
#     argument_list = []
#     if train_flag:
#         argument_list.extend([
#             A.Resize(height=image_size[0], width=image_size[1], p=1.0),
#             A.RandomResizedCrop(height=image_size[0], width=image_size[1], 
#                                scale=(0.5, 1.0), ratio=(0.75, 1.3333333333333333), 
#                                interpolation=cv2.INTER_LINEAR, p=1.0),
            
#             A.ShiftScaleRotate(shift_limit=(-0.05, 0.05), scale_limit=(-0.05,0.05),
#                                rotate_limit=10, border_mode=0 , value=0, p=0.3),
                                                            
            
           
#             A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0, hue=0, p=0.5),
            
#         ])
#     else:
#         argument_list.extend([A.Resize(height=image_size[0], width=image_size[1], p=1.0),
                           
#                              ])
#     print(argument_list)
#     return A.Compose(argument_list)


import torchvision.transforms as transforms
import torch
import numpy as np

class JointTransform:
    def __init__(self, image_size, train_flag=True):
        self.train_flag = train_flag
        self.resize = transforms.Resize((image_size[0], image_size[1]))

        if train_flag:
            self.img_transform = transforms.Compose([
                transforms.ToPILImage(),
                self.resize,
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.ToTensor(),
            ])
        else:
            self.img_transform = transforms.Compose([
                transforms.ToPILImage(),
                self.resize,
                transforms.ToTensor(),
            ])

    def __call__(self, image, mask):
        image = self.img_transform(image)

        mask = torch.from_numpy(mask).long()
        # Convert to tensor if not already
        if not isinstance(mask, torch.Tensor):
            mask = torch.from_numpy(mask)

        # If mask is H x W x 1 → convert to H x W
        if mask.ndim == 3 and mask.shape[-1] == 1:
            mask = mask.squeeze(-1)

        # Now mask should be H x W
        # Add batch + channel dimensions
        mask = mask.unsqueeze(0).unsqueeze(0)  # (1,1,H,W)

        mask = torch.nn.functional.interpolate(
            mask.float(),
            size=(1024, 1024),
            mode='nearest'
        )

        mask = mask.squeeze(0)  # remove batch dim
        mask = torch.nn.functional.interpolate(
            mask.unsqueeze(0).unsqueeze(0).float(),
            size=(image.shape[1], image.shape[2]),
            mode="nearest"
        ).squeeze(0).squeeze(0).long()

        return {"image": image, "mask": mask}


def get_augmentation_gray(image_size, train_flag=True):
    return JointTransform(image_size, train_flag)