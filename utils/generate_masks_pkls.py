import os
import torch
import torchvision.transforms as transforms
from PIL import Image
import pickle
from tqdm import tqdm
dataset_dir = '../' # folder path of Butterfly and CAMUS

transform = transforms.ToTensor()

train_dir = os.path.join(dataset_dir, 'train')
masks_dir = os.path.join(dataset_dir, 'masks')
os.makedirs(masks_dir, exist_ok=True)
pkl_file_indexes = []

for sub_dir in tqdm(os.listdir(train_dir)):
    # traverse all subdirectories
    subdir_path = os.path.join(train_dir, sub_dir)
    if os.path.isdir(subdir_path):
        label_dir = os.path.join(subdir_path, 'label')

        if os.path.isdir(label_dir):
            # ensure is a directory
            for filename in os.listdir(label_dir):
                if filename.endswith('.png'):
                    file_path = os.path.join(label_dir, filename)

                    # load image and convert to tensor
                    image = Image.open(file_path)
                    transform = transforms.ToTensor()
                    tensor = transform(image)[0]

                    # create pickle file path for mask image
                    pkl_file = f'{sub_dir}_{filename.split(".")[0]}_gt.pkl'
                    pkl_file_path = os.path.join(masks_dir, pkl_file)

                    # save the pkl file
                    with open(pkl_file_path, 'wb') as f:
                        pickle.dump(tensor, f)

                    # add to indexes
                    img_path = os.path.join(subdir_path, "img", filename)
                    if not os.path.exists(img_path):
                        continue

                    pkl_file_indexes.append([img_path, pkl_file_path])

# add all the pkl files indexes to the list
pkl_file_indexes_path = os.path.join(dataset_dir, 'train_tf_img_to_gt.pkl')
with open(pkl_file_indexes_path, 'wb') as f:
    pickle.dump(pkl_file_indexes, f)

print("Done!")