import glob
import os
import json
import re
import PIL.Image as Image
import PIL.ImageStat as ImageStat
from collections import defaultdict
from tqdm import tqdm
from scipy.io.matlab import loadmat
from SUNRGBDtoolbox_python.SUNRGBD import readFrame
import random as rand

# all_ds_files = glob.glob('./SUNRGBD/**', recursive=True)
#
# image_names = [iname for iname in all_ds_files if 'image/' in iname]
# img_numbs = sorted([i.split('/')[-1].split('.jpg')[0] for i in image_names])
#
# d2_anno_names = [iname for iname in all_ds_files if 'annotation2Dfinal/' in iname and not 'json_' in iname]
# d3_anno_names = [iname for iname in all_ds_files if 'annotation3Dfinal/' in iname and not 'json_' in iname]
# d2_3_anno_names = [iname for iname in all_ds_files if 'annotation2D3D/' in iname and not 'json_' in iname]
# anno_types = {
#     '2D': d2_anno_names,
#     '3D': d3_anno_names,
#     '2D3D': d2_3_anno_names
# }


def mean_pixel_level(img_file):
    img = Image.open(img_file).convert('L')
    img_stats = ImageStat.Stat(img)
    return img_stats.mean[0]


def compute_img_brightnesses(image_files):
    img_brightnesses = {}
    for img in image_files:
        img_brightnesses[img] = mean_pixel_level(img)
    return img_brightnesses


def bbox_from_poly(polygon):
    bbox = None
    return bbox


def build_v2_addition(v2_matlab_arr):
    v2_anno = {}
    for i in range(v2_matlab_arr.shape[0]):
        try:
            image_name = v2_matlab_arr['sequenceName'][i].item()[0].split('/')[-1]
            bboxes = v2_matlab_arr['groundtruth2DBB'][i][0][0]
            v2_anno[image_name] = {
                'objects': {
                    bboxes[j]['objid'].item(): 
                    {
                        'classname': bboxes[j]['classname'].item(),
                        'has3D': bboxes[j]['has3dbox'].item(), 
                        'rect': bboxes[j]['gtBb2D']
                    } 
                    for j in range(bboxes.shape[0])
                }
            }
        except IndexError as e:
             v2_anno[image_name] = {}
    return v2_anno


def build_v2_addition_3d(v2_matlab_arr):
    v2_anno = {}
    for i in range(v2_matlab_arr.shape[0]):
        try:
            image_name = v2_matlab_arr['sequenceName'][i].item()[0].split('/')[-1]
            bboxes = v2_matlab_arr['groundtruth3DBB'][i][0][0]
            print(bboxes.dtype.names)
            print(bboxes)
            v2_anno[image_name] = {
                'objects': {
                    bboxes[j]['objid'].item(): 
                    {
                        'classname': bboxes[j]['classname'].item(),
                        '3D_coords': bboxes[j]['gtCorner3D']
                    } 
                    for j in range(bboxes.shape[0])
                }
            }
        except IndexError as e:
             v2_anno[image_name] = {}
    return v2_anno


def read_sun_dataset(all_2d_files):



    base_paths = [fp.split('annotation2Dfinal')[0] for fp in all_2d_files]

    anno_types_to_load = {
        '2D': 'annotation2Dfinal/',
        # '3D': 'annotation3Dfinal/',
        # '2D3D': 'annotation2D3D/'
    }

    sun_rgb_anno = defaultdict(lambda: defaultdict(dict))
    for base_path in tqdm(base_paths):
        img_dir = base_path.split('/')[-2]
        for ann_type, anno_dir in anno_types_to_load.items():
            try:
                with open(os.path.join(base_path, anno_dir, 'index.json'), 'r') as f:
                    img_annotation = json.load(f)
            except (json.JSONDecodeError, FileNotFoundError) as e:
                continue
            sun_rgb_anno[img_dir]['objects'][ann_type] = img_annotation

        with open(os.path.join(base_path, 'scene.txt'), 'r') as f:
            scene = f.read()

        with open(os.path.join(base_path, 'intrinsics.txt'), 'r') as f:
            intrinsics = f.read().splitlines()

        # image_segmentation = loadmat(os.path.join(base_path, 'seg.mat'))

        sun_rgb_anno[img_dir]['intrinsics'] = intrinsics
        sun_rgb_anno[img_dir]['scene'] = scene
        # sun_rgb_anno[img_dir]['segmentation'] = image_segmentation
        sun_rgb_anno[img_dir]['imgPath'] = '/'.join(base_path.split('/')[:-2])

    return {k: dict(v) for k, v in sun_rgb_anno.items()}


def build_dataset(sun_dataset):
    ds_scaffold = {}
    for image_name, annotations in sun_dataset.items():
        new_image_entry = {
            'imageID': image_name,
            "sunPath": annotations['imgPath'],
            "scene": annotations['scene']
        }
        ds_scaffold[image_name] = new_image_entry
    return ds_scaffold


"""
design draft v3:
{
    image: {
        name
        scene
        layout
        intrinsics
        detector
        filenames: {
            image
            fullres
            depth
        }
        segmentation_arr
        objects : {
            obj_id: {
                label
                globalID
                2D: {
                    polygon: []
                    rectangle: [
                },
                3D: {
                    depth
                },
                phys_props: {

                }
            }
            .
            .
            .
        }
    }
}
"""