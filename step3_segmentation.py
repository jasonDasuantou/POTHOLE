import os
from options.ours_options import OursOptions
from data import CreateDataLoader
from models import create_model
from util.util import save_images
import torch
import numpy as np

if __name__ == '__main__':
    opt = OursOptions().parse()
    opt.num_threads = 1
    opt.batch_size = 1
    opt.serial_batches = True  # no shuffle
    opt.isTrain = False

    save_dir = opt.results_dir
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    data_loader = CreateDataLoader(opt)
    dataset = data_loader.load_data()
    model = create_model(opt, dataset.dataset)
    model.setup(opt)
    model.eval()

    test_loss_iter = []
    epoch_iter = 0
    conf_mat = np.zeros((dataset.dataset.num_labels, dataset.dataset.num_labels), dtype=np.float32)
    with torch.no_grad():
        for i, data in enumerate(dataset):

            model.set_input(data)
            model.forward()
            epoch_iter += opt.batch_size
            print(epoch_iter)
            path_ = 'result/'

            save_images(path_, model.get_current_visuals(), model.get_image_names())

