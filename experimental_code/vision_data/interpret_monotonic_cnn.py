import PIL.Image
import torch

from mononet.monotonic_cnn import *
from mononet.datasets.vision_data import get_data, MEDMNIST_INVERSE_TRANSFORM
import click
import math
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
import numpy as np
from PIL import ImageDraw, ImageEnhance


def highlight_area(img, region, factor, outline_color=None, outline_width=1):
    """ Highlight specified rectangular region of image by `factor` with an
        optional colored  boarder drawn around its edges and return the result.
    """
    img = img.copy()  # Avoid changing original image.
    img_crop = img.crop(region)

    brightner = ImageEnhance.Brightness(img_crop)
    img_crop = brightner.enhance(factor)

    img.paste(img_crop, region)

    # Optionally draw a colored outline around the edge of the rectangular region.
    if outline_color:
        draw = ImageDraw.Draw(img)  # Create a drawing context.
        left, upper, right, lower = region  # Get bounds.
        coords = [(left, upper), (right, upper), (right, lower), (left, lower),
                  (left, upper)]
        draw.line(coords, fill=outline_color, width=outline_width)

    return img


def load_model_and_data(model_name, model_path, data_path, data_flag, batch_size):
    ds, loaders, info = get_data(data_flag, data_path, batch_size)
    model = globals()[model_name].load_from_checkpoint(model_path)
    model.eval()
    return model, ds, loaders, info


def pytorch_batch_one_to_pil(sample, idx=0):
    sample_pil = MEDMNIST_INVERSE_TRANSFORM(sample[idx])
    sample_rgb = PIL.Image.new("RGBA", sample_pil.size)
    sample_rgb.paste(sample_pil)
    return sample_rgb


def make_feat_maps_grid(feat_maps):
    maps = feat_maps.transpose(0, 1)
    nrow = math.ceil(math.sqrt(maps.shape[0]))
    grid = make_grid(maps, nrow=nrow, normalize=True, scale_each=True, pad_value=0, padding=4)
    return grid


def topk_1dindex_to_2dindex(positions, feats):
    curr_pos = positions[0]
    curr_feats = feats[0]
    pos2d = []
    for idx, channel in enumerate(curr_pos):
        pos2d.append(np.array(np.unravel_index(channel.numpy(), curr_feats[idx].shape)).T)
    return np.stack(pos2d, axis=0)


@click.command()
@click.option('--model_name', default='ResidualMonotonicCNN1', help="Model to interpret")
@click.option('--data_path', default='.', help="Path where to store/load from the data")
@click.option('--data_flag', default='dermamnist', help="Dataset to interpret")
@click.option('--model_path', required=True, help="Path to the model checkpoint")
@click.option('--data_idx', default=0, help="Index of the datapoint to interpret")
@click.option('--results_path', required=True, help="Where to store the interpretability results")
def interpret(model_name, data_path, data_flag, model_path, data_idx, results_path):
    BATCH_SIZE = 64
    model, ds, loaders, info = load_model_and_data(model_name, model_path, data_path, data_flag, BATCH_SIZE)

    dataset = ds['test']

    sample = dataset[data_idx][0].unsqueeze(dim=0)
    feature_maps = model.get_feature_maps(sample)
    print(torch.flatten(feature_maps, start_dim=2).shape)
    max_value, max_ids = torch.max(torch.flatten(feature_maps, start_dim=2), dim=2)
    map_max_norm_idx = torch.argmax(max_value)
    interpretable_feats, position = model.get_features(sample)
    pred = torch.argmax(model(sample), dim=1)
    print(map_max_norm_idx)
    print(f"Prediction: {info['label'][str(int(pred[0]))]}")
    positions2d = topk_1dindex_to_2dindex(position, feature_maps)

    IDX = 16
    K = 0

    try:
        conv_filters = model.feat1.weight
        padding = model.feat1.padding
        ks = model.feat1.kernel_size
        alpha = model.interpret.weight[IDX*model.topk + K]
        beta = model.interpret_out.weight[int(pred[0])]
        direction = float(alpha*beta)
        print(direction)
    except:
        raise RuntimeError("Filters not found!")

    location_h, location_w = positions2d[IDX][K]
    sample_pil = pytorch_batch_one_to_pil(sample)
    filter_pil = pytorch_batch_one_to_pil(conv_filters, idx=IDX)
    feature_maps_grid = pytorch_batch_one_to_pil(make_feat_maps_grid(feature_maps).unsqueeze(0))
    filter_grid = pytorch_batch_one_to_pil(make_feat_maps_grid(conv_filters.transpose(0, 1)).unsqueeze(0))
    sample_with_patch = highlight_area(
        sample_pil,
        (
            max(location_w - padding[1], 0),
            max(location_h - padding[0], 0),
            min(location_w - padding[1] + ks[1], 27),
            min(location_h - padding[0] + ks[0], 27),
        ),
        1.,
        'red'
    )
    plt.figure(1)
    plt.imshow(sample_pil)
    plt.axis('off')
    plt.figure(2)
    plt.imshow(filter_grid)
    plt.axis('off')
    plt.figure(3)
    plt.imshow(filter_pil)
    plt.axis('off')
    plt.figure(4)
    plt.imshow(sample_with_patch)
    plt.axis('off')
    plt.show()


if __name__ == '__main__':
    interpret()
