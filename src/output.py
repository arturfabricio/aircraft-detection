import torch 
from torch.utils.data import Dataset
from main import *
import torch.nn as nn

dir_root = Path(__file__).parent.parent
model_dir = Path(dir_root, './data/model/')
train_imgs = Path(dir_root, './data/train')
test_imgs = Path(dir_root, './data/test')

bboxs = bbox_utils.generate(10, 130//4, 10, (128, 128))

model = torch.load(model_dir)
model.eval()

out_data = model(img)