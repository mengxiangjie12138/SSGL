import torchvision.transforms as transforms

class_num = 4

# resnet
train_batch_size = 70
test_batch_size = 70


# GCN_model
features_dim_num = 2048
GCN_hidderlayer_dim_num = 512


# train_GCN
train_dataset_path = 'MicroData/train'
train_dataset_label_path = 'MicroData/with_label'
train_dataset_non_label_path = 'MicroData/non_label'
test_dataset_path = 'MicroData/test'

epoches = 500
k = 10

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])











