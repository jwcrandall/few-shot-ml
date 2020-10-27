import os
import random
import shutil
import torch
import pathlib
import math
import csv

from nshot.sampler import ShotWaySampler

import numpy as np
import copy 

import matplotlib.pyplot as plt
from matplotlib.pyplot import cm


from PIL import Image

from torchvision import transforms

data_dir = pathlib.Path(
    '/home/shared/'
)

def visualize(episode, query_rows=None, scale=3):

    support = episode['support']
    support_labels = episode['support_labels']
    query = episode['query']

    labels = list(set(support_labels))

    num_classes = len(labels)
    max_images = max(
        [
            len(
                [label for label in support_labels if label == class_name]
            ) for class_name in labels
        ]
    )

    num_queries = len(query)
    if query_rows is None:
        query_rows = math.ceil(num_queries / max_images)

    fig, ax = plt.subplots(
        num_classes + query_rows,
        max_images,
        figsize=(scale * max_images, scale * (num_classes + query_rows))
    )

    for row_index, class_name in enumerate(labels):
        color = cm.rainbow(np.linspace(0, 1, num_classes))

        indices = [
            i for i, label in enumerate(support_labels) if label == class_name
        ]

        images = [support[i] for i in indices]

        for col_index, image in enumerate(images):

            c = color[row_index]

            ax[row_index][col_index].imshow(image)
            ax[row_index][col_index].set_xticks([])
            ax[row_index][col_index].set_yticks([])

            for axis in ['top', 'bottom', 'left', 'right']:
                ax[row_index][col_index].spines[axis].set_linewidth(3)
                ax[row_index][col_index].spines[axis].set_color(c)

    count = 0
    for index, image in enumerate(query):
        count += 1

        col_index = math.floor(index / query_rows)
        row_index = index % query_rows + num_classes

        ax[row_index][col_index].imshow(image)
        ax[row_index][col_index].set_xticks([])
        ax[row_index][col_index].set_yticks([])

        for axis in ['top', 'bottom', 'left', 'right']:
            ax[row_index][col_index].spines[axis].set_linewidth(5)
            ax[row_index][col_index].spines[axis].set_color('black')

        if count == query_rows * max_images:
            break

    plt.show()

    return None


def get_episode(support_dir, query_dir='cifar_sample', preprocess_images=True):

    support, support_classes = load_set(
        support_dir,
        preprocess_images=preprocess_images
    )

    query, _ = load_set(
        query_dir,
        preprocess_images=preprocess_images
    )

    names2labels = {
        name: f'S{i}' for i, name in enumerate(set(support_classes))
    }

    support_labels = [names2labels[name] for name in support_classes]

    episode = {
        'support': support,
        'support_labels': support_labels,
        'query': query
    }

    return episode, list(set(support_classes))


def load_set(directory, preprocess_images=True):
    directory = pathlib.Path(directory)

    datapoint_set = []
    datapoint_classes = []

    for class_name in os.listdir(directory):
        class_dir = directory / class_name

        for filename in os.listdir(class_dir):
            
            if filename.startswith('.'):
                continue

            filepath = class_dir / filename

            datapoint = load(filepath)

            if preprocess_images:
                datapoint = preprocess(datapoint)

            datapoint_set.append(
                datapoint
            )

            datapoint_classes.append(
                class_name
            )

    return datapoint_set, datapoint_classes


def load(filepath):

    datapoint = Image.open(filepath).convert('RGB')
    datapoint.load()

    return datapoint


def preprocess(datapoint):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    operations = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    datapoint = operations(datapoint)
    return datapoint


def write_csv(data, name):
    if not os.path.exists('splits'):
        os.makedirs('splits')
    with open(f'./splits/{name}.csv','w') as myfile:
        wr = csv.writer(myfile) #, quoting=csv.QUOTE_ALL)
        wr.writerows(data)
    return pathlib.Path(f'./splits/{name}.csv')


def reset_data(shots=5, ways=5, seed=1234):

    print('Resetting data...', flush=True, end=' ')
    random.seed(seed)

    if shots > 8:
        raise ValueError('Images per class must be less than 9.')

    if ways > 10:
        raise ValueError('Number of classes must be less than 11.')

    # Delete cifar sample directory
    if os.path.exists('cifar_sample'):
        shutil.rmtree('cifar_sample')

    # Delete episode directory
    if os.path.exists('episode'):
        shutil.rmtree('episode')

    # Resample cifar sample directory
    for class_name in os.listdir(data_dir / 'CIFAR10/test'):
        directory = data_dir / f'CIFAR10/test/{class_name}'
        images = os.listdir(directory)
        images = random.sample(images, 8)

        image_paths = [(directory / image, image) for image in images]

        os.makedirs(f'cifar_sample/{class_name}')

        for filepath, name in image_paths:
            shutil.copy(filepath, f'cifar_sample/{class_name}/{name}')

    # Resample episode
    classes = random.sample(os.listdir('cifar_sample'), ways)

    for class_name in classes:
        directory = pathlib.Path('cifar_sample') / class_name
        images = os.listdir(directory)
        images = random.sample(images, shots)

        image_paths = [(directory / image, image) for image in images]

        os.makedirs(f'episode/{class_name}')

        for filepath, name in image_paths:
            shutil.copy(filepath, f'episode/{class_name}/{name}')

    print('Done.', flush=True)


def load_weights(weights_file, model):
    sd = torch.load(weights_file)
    sd_final = {}
    for k, v in sd.items():
        if k[:7] == 'module.':
            sd_final[k[7:]] = v
        else:
            sd_final[k] = v
    model.load_state_dict(sd_final)


def visualize_predictions(episode, logits, num_queries=5, support_classes=None,
                          scale=3, random_order=False):

    query = episode['query']

    if support_classes is None:
        support_classes = list(set(episode['support_labels']))
        no_class_names = True
    else:
        no_class_names = False

    fig, ax = plt.subplots(
        num_queries,
        2,
        figsize=(scale * 2, scale * num_queries)
    )
    index_list = [i for i in range(len(query))]

    if random_order:
        random.shuffle(index_list)

    for row_index in range(num_queries):

        image_index = index_list[row_index]

        image = query[image_index]
        probabilities = torch.softmax(logits[image_index], axis=0)
        probabilities = np.around(probabilities.detach().numpy(), decimals=2)

        ax[row_index][0].imshow(image)
        ax[row_index][0].set_xticks([])
        ax[row_index][0].set_yticks([])

        for axis in ['top', 'bottom', 'left', 'right']:
            ax[row_index][0].spines[axis].set_linewidth(3)
            ax[row_index][0].spines[axis].set_color('black')

        ax[row_index][1].set_axis_off()

        message = ''

        for class_index, class_name in enumerate(support_classes):
            if no_class_names:
                message += f'{class_name}: {probabilities[class_index]} \n'
            else:
                prob = probabilities[class_index]
                message += f'S{class_index} - {class_name}: {prob} \n'

        message += f'Index: {image_index}'

        ax[row_index][1].text(
            0, 0,
            message
        )

    plt.show()


def get_raw_episode(dataset, num_classes, num_datapoints, num_queries,
                    distractors=False):
    """get_episode will grab an episode using the shotway sampler

    :param dataset: The dataset the user wants to grab an episode from
    :type dataset: NShotDataset
    :param num_classes: The classes each episode has
    :type num_classes: list of ints
    :param num_datapoints: The number of examples per class
    :type num_datapoints: list of ints
    :param num_queries: The batch size used
    :type num_queries: int
    :param distractors: Indicates whether the user wants to use distractors
    :type distractors: boolean, defaults to False
    :return: Returns an episode
    :rtype: tuple
    """
    sampler = ShotWaySampler(
        dataset.labels,
        1,
        num_classes,
        num_datapoints,
        num_queries,
        distractors=distractors
    )

    sampler_iter = iter(sampler)
    episode = [dataset[idx] for idx in next(sampler_iter)]

    return episode


def conver_to_img(episode):
    episode_out = {}
    support = []
    support_labels = []
    query = []
    labels = []
    
    for point in episode:
        if not 'S' in point[2]:
            query.append(load(point[0]))
            labels.append(point[1])
        else:
            support .append(load(point[0]))
            support_labels.append(point[1])
            
    episode_out['support'] = support
    episode_out['support_labels'] = support_labels
    episode_out['query'] = query

    return episode_out


def visualize_from_dataset(dataset):
    new_dataset = copy.deepcopy(dataset) 
    new_dataset.lazy = False
    def identity(x):
        return(x)

    new_dataset.augmentor = identity
    episode = get_raw_episode(new_dataset, num_classes = 5, num_datapoints = 5, num_queries = 0, distractors=False)
    episode = conver_to_img(episode)
    visualize(episode)
    

def load_checkpoint(model, checkpoint_number):
    checkpoints_dir = "/home/shared/weights/weights_for_demo/logs/weights"
    checkpoints_path = pathlib.Path(checkpoints_dir)
    checkpoints = os.listdir(checkpoints_dir)
    
    if checkpoint_number >= len(checkpoints):
        print("Check point does not exisit")
    else:
        load_weights(checkpoints_path / checkpoints[checkpoint_number], model)
