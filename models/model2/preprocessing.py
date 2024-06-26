import torchvision
from torchvision import transforms as T

preprocessing_func = T.Compose(
    [T.Resize((224, 224)),
     T.ToTensor(),
     T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
     ]
)

def preprocess(img):
    return preprocessing_func(img)