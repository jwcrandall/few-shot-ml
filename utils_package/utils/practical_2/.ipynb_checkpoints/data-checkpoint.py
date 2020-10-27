import nshot
import pathlib
import torch
import tqdm
import os

from torchvision import transforms

from ..main import preprocess
from ..main import load


def get_omniglot(train=True, num_episodes=10, large_images=False):

    data_dir = pathlib.Path(
        '/qfs/projects/unicornface/benchmark_datasets/'
    )

    if not os.path.exists(data_dir):
        data_dir = pathlib.Path(
            '/home/shared/omniglot/'
        )

    if train:
        csv_path = data_dir / 'splits/omniglot/train.csv'
    else:
        csv_path = data_dir / 'splits/omniglot/val.csv'

    if large_images:
        dataset = nshot.load_data(
            data_dir / 'uncompressed/omniglot',
            csv_path=csv_path,
            load_type='csv',
            preprocess_fn=preprocess,
            datapoint_loader=load,
            augmentor=lambda x: x
        )
    else:
        dataset = nshot.load_data(
            data_dir / 'uncompressed/omniglot',
            csv_path=csv_path,
            load_type='csv',
            preprocess_fn=small_preprocess,
            datapoint_loader=load,
            augmentor=lambda x: x
        )

    sampler = nshot.sampler.ShotWaySampler(
        dataset.labels,
        episodes_per_iteration=num_episodes,
        classes_per_episode=5,
        positives_per_class=5,
        query_size=4,
        distractors=True
    )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=2,
        collate_fn=nshot.run_.utils.episodic_collate,
        batch_sampler=sampler
    )

    return tqdm.tqdm(iter(dataloader), ncols=100)


def small_preprocess(datapoint):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    operations = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    datapoint = operations(datapoint)

    return datapoint
