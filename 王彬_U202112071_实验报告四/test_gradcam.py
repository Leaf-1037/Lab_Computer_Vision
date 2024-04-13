import math
import torch
import numpy as np
from torch import Tensor
from torch import nn
import torch.nn.functional as F
from typing import Optional, List
import torchvision.transforms as transforms
from PIL import Image
import torchvision.models as models
from torch import Tensor
from matplotlib import cm
from torchvision.transforms.functional import to_pil_image

import matplotlib.pyplot as plt

img_path = './data4/both.jpg' 
save_path = './output/both_category_cat_gradcam.png'   

preprocess = transforms.Compose([transforms.Resize(256),
                                transforms.CenterCrop(224),
                                transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

net = torch.load('torch_alex.pth')   # 导入模型
print(net)

# 建立列表容器，用于盛放输出特征图
feature_map = []

def forward_hook(module, inp, outp): 
    feature_map.append(outp)  

# 对最后一层进行前向传播
net.features.register_forward_hook(forward_hook)  

# 特征图梯度列表
grad = [] 

def backward_hook(module, inp, outp): 
    grad.append(outp) 

net.features.register_full_backward_hook(backward_hook)    # 对net.features这一层注册反向传播

print(grad)

# 图片预处理并转换为RGB模型
orign_img = Image.open(img_path).convert('RGB')
img = preprocess(orign_img)
img = torch.unsqueeze(img, 0) 

out = net(img)
print(out)
# out = net(img.cuda())     # 前向传播 
# 获取预测类别编码
# cls_idx = torch.argmax(out).item() 
# cls_idx = 0 : cat
# cls_idx = 1 : dog
cls_idx = 0
# 获取预测类别分数
score = out[:, cls_idx].sum()
# 由预测分数反向传播得到梯度
net.zero_grad() 
score.backward(retain_graph=True)

# 将各层梯度取平均池化
weights = grad[0][0].squeeze(0).mean(dim=(1, 2)) 

plt.subplots(16,16,figsize=(32,32),dpi=100)

# 对特征图的通道进行加权叠加
grad_cam = (weights.view(*weights.shape, 1, 1) * feature_map[0].squeeze(0)).sum(0)
# print(weights.view(*weights.shape, 1, 1) * feature_map[0].squeeze(0))


def _normalize(cams: Tensor) -> Tensor:
        """CAM normalization"""
        cams.sub_(cams.flatten(start_dim=-2).min(-1).values.unsqueeze(-1).unsqueeze(-1))
        cams.div_(cams.flatten(start_dim=-2).max(-1).values.unsqueeze(-1).unsqueeze(-1))

        return cams


    


grad_cam = _normalize(F.relu(grad_cam, inplace=True)).cpu()
mask = to_pil_image(grad_cam.detach().numpy(), mode='F')

def overlay_mask(img: Image.Image, mask: Image.Image, colormap: str = 'jet', alpha: float = 0.6) -> Image.Image:
    """Overlay a colormapped mask on a background image

    Args:
        img: background image
        mask: mask to be overlayed in grayscale
        colormap: colormap to be applied on the mask
        alpha: transparency of the background image

    Returns:
        overlayed image
    """

    if not isinstance(img, Image.Image) or not isinstance(mask, Image.Image):
        raise TypeError('img and mask arguments need to be PIL.Image')

    if not isinstance(alpha, float) or alpha < 0 or alpha >= 1:
        raise ValueError('alpha argument is expected to be of type float between 0 and 1')

    cmap = cm.get_cmap(colormap)    
    # Resize mask and apply colormap
    overlay = mask.resize(img.size, resample=Image.BICUBIC)
    overlay = (255 * cmap(np.asarray(overlay) ** 2)[:, :, 1:]).astype(np.uint8)
    # Overlay the image with the mask
    overlayed_img = Image.fromarray((alpha * np.asarray(img) + (1 - alpha) * overlay).astype(np.uint8))

    return overlayed_img

# 对每一个通道进行normalize后，绘制每个通道的子图，汇总到16*16的大图中
list_channel = weights.view(*weights.shape, 1, 1) * feature_map[0].squeeze(0)
for i in range(list_channel.size(0)):
    imt = list_channel[i]
    plt.subplot(16,16,i+1)
    _grad_sub = _normalize(F.relu(imt, inplace=True)).cpu()
    mask1 = to_pil_image(_grad_sub.detach().numpy(), mode='F')
    result1 = overlay_mask(orign_img, mask1)
    plt.imshow(result1)

result = overlay_mask(orign_img, mask) 
result.show()
# plt.show()
# plt.savefig('./output/dog_gradcam_each_channel.png')
# plt.show()
result.save(save_path)