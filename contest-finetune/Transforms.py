from torchvision import transforms


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

IMAGE_SIZE = 224


def get_training_transform():
    transform = transforms.Compose([
        transforms.ToTensor(),          
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE), antialias=True),  # Resize the image to IMAGE_SIZE*IMAGE_SIZE
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop((IMAGE_SIZE, IMAGE_SIZE), padding=(12, 12), padding_mode='edge'),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        transforms.RandomErasing(scale=(0.02, 0.4), value=IMAGENET_MEAN)
    ])

    return transform


def get_test_transform(equalized=False):

    if equalized:
        return transforms.Compose([
            transforms.RandomEqualize(p=1),        
            transforms.ToTensor(),  
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE), antialias=True),  # Resize the image to IMAGE_SIZE*IMAGE_SIZE
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    else:
        return transforms.Compose([
            transforms.ToTensor(),          
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE), antialias=True),  # Resize the image to IMAGE_SIZE*IMAGE_SIZE
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
