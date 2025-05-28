import random
import glob

random.seed(42)

trainval_file = '../data/VOCdevkit/VOC2012/ImageSets/Segmentation/trainval.txt'
all_image_ids = set()
with open(trainval_file, 'r') as f:
    lines = f.readlines()

image_ids = [line.strip().split()[0] for line in lines if line.strip()]
all_image_ids.update(image_ids)

image_ids = list(all_image_ids)
random.shuffle(image_ids)

total = len(image_ids)
train_size = int(total * 0.7)  
val_size = int(total * 0.15)   
test_size = total - train_size - val_size  

# split the data into train, val, and test sets
train_lines = [f"{id}\n" for id in image_ids[:train_size]]
val_lines = [f"{id}\n" for id in image_ids[train_size:train_size + val_size]]
test_lines = [f"{id}\n" for id in image_ids[train_size + val_size:]]

with open('../data/VOCdevkit/VOC2012/split/train.txt', 'w') as f:
    f.writelines(train_lines)
with open('../data/VOCdevkit/VOC2012/split/val.txt', 'w') as f:
    f.writelines(val_lines)
with open('../data/VOCdevkit/VOC2012/split/test.txt', 'w') as f:
    f.writelines(test_lines)

print(f"Total images: {total}")
print(f"Train: {len(train_lines)}, Val: {len(val_lines)}, Test: {len(test_lines)}")