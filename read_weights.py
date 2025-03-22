from deepforest import main
from torchvision.models import resnet50

model_release = main.deepforest()
model_release.use_release()
backbone = model_release.model.backbone
print("backbone     ", backbone)
resnet_weights = backbone.body.state_dict()

import pickle

with open('resnet50_weights.pkl','wb') as f:
    pickle.dump(resnet_weights, f)
