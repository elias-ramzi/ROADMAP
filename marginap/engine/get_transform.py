import torchvision.transforms as T


def Identity():

    def __call__(self, img):
        return img


def get_train_transform(with_resize=True):
    return T.Compose([
            T.Resize(256) if with_resize else Identity(),
            T.RandomResizedCrop(224, scale=(0.16, 1), ratio=(0.75, 1.33)),
            T.RandomHorizontalFlip(0.5),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])


def get_test_transform():
    return T.Compose([
            T.Resize((256, 256)),
            T.CenterCrop((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
