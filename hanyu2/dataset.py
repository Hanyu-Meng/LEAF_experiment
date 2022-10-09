import tensorflow as tf
import tensorflow_datasets as tfds
import data
import config as CONF

def prepare(dataset,info):
    dataset = data.prepare(dataset, batch_size=CONF.BATCH_SIZE)
    # n_of_class = info.features['instrument']['family'].num_classes
    # n_of_class = info.features['pitch'].num_classes
    n_of_class = info.features['label'].num_classes
    # n_of_class = info.features['speaker_id'].num_classes
    return dataset, n_of_class

def load(dataset_name):
    if dataset_name == 'voxforge':
        dataset, info = tfds.load("voxforge", split='train', shuffle_files=True, data_dir="./dataset/downloads/voxforge",with_info=True)
        print("LOADED")
        print(dataset)
    else:
        dataset,info = tfds.load(dataset_name,with_info=True,data_dir="./dataset")
    dataset, n_of_class = prepare(dataset,info)
    return dataset, n_of_class