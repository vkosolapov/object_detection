import yaml
from ..yolov5.utils.datasets import create_dataloader


def create_train_dataloader(data_path, image_size, batch_size, workers):
    data_dict = yaml.safe_load(data_path)
    train_path = data_dict['train']
    train_loader = create_dataloader(
        train_path, 
        image_size, 
        batch_size, 
        gs=32, 
        single_cls=False,
        # hyp=hyp, 
        augment=False, 
        cache=None, 
        rect=False, 
        rank=-1,
        workers=workers,
        prefix="train", 
        shuffle=True
    )[0]
    return train_loader

def create_val_dataloader(data_path, image_size, batch_size, workers):
    data_dict = yaml.safe_load(data_path)
    val_path = data_dict['val']
    val_loader = create_dataloader(
        val_path, 
        image_size, 
        batch_size, 
        gs=32, 
        single_cls=False,
        # hyp=hyp, 
        augment=False, 
        cache=None, 
        rect=False, 
        rank=-1,
        workers=workers,
        prefix="val"
    )[0]
    return val_loader

def rename(file_name, string_to_add):
    with open(file_name, 'r') as f:
        file_lines = [''.join([string_to_add, x.strip(), '\n']) for x in f.readlines()]
    with open(file_name, 'w') as f:
        f.writelines(file_lines)


if __name__ == "__main__":
    rename("data/AFO/PART_1/PART_1/train.txt", "data/AFO/PART_1/PART_1/images/")
    rename("data/AFO/PART_1/PART_1/validation.txt", "data/AFO/PART_1/PART_1/images/")
    rename("data/AFO/PART_1/PART_1/test.txt", "data/AFO/PART_1/PART_1/images/")
