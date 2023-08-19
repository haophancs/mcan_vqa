import h5py
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.utils.data
import torchvision.models as models
from tqdm import tqdm
from transformers import AutoModel

import config
from data_utils.utils import get_transform
from data_utils.vqa_image import VQAImages


def create_vqa_loader(path, extension='png'):
    transform = get_transform(config.image_size)
    dataset = VQAImages(path, extension=extension, transform=transform)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config.preprocess_batch_size,
        num_workers=config.data_workers,
        shuffle=False
    )
    return data_loader


def main():
    cudnn.benchmark = True

    net = AutoModel.from_pretrained("microsoft/resnet-50").cuda()
    net.eval()

    train_loader = create_vqa_loader(config.train_path, config.image_extension)
    test_loader = create_vqa_loader(config.test_path, config.image_extension)
    features_shape = (
        len(train_loader.dataset) + len(test_loader.dataset),
        config.output_features,
        config.output_size,
        config.output_size
    )

    with h5py.File(config.preprocessed_path, 'w', libver='latest') as fd:
        features = fd.create_dataset('features', shape=features_shape, dtype='float16')
        image_ids = fd.create_dataset(
            'ids',
            shape=(len(train_loader.dataset) + len(test_loader.dataset),),
            dtype='int32'
        )

        i = j = 0
        for ids, imgs in tqdm(train_loader):
            imgs = imgs.clone().cuda()
            with torch.no_grad():
                out = net(imgs).last_hidden_state
                out = out.detach().cpu()

            j = i + imgs.size(0)
            features[i:j, :, :] = out.numpy().astype('float16')
            image_ids[i:j] = ids.numpy().astype('int32')
            i = j

            del imgs
            torch.cuda.empty_cache()

        for ids, imgs in tqdm(test_loader):
            imgs = imgs.clone().cuda()
            with torch.no_grad():
                out = net(imgs).last_hidden_state
                out = out.detach().cpu()

            j = i + imgs.size(0)
            features[i:j, :, :] = out.numpy().astype('float16')
            image_ids[i:j] = ids.numpy().astype('int32')
            i = j

            del imgs
            torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
