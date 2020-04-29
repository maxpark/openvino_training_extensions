# Copyright (C) 2020 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions
# and limitations under the License.

""" Converts WiderPerson annotation to COCO format. """

import json
from collections import defaultdict
import os

import argparse
import imagesize
from tqdm import tqdm


def parse_args():
    """ Parses input arguments. """

    parser = argparse.ArgumentParser(description='Convert dataset')
    parser.add_argument('input_annotation',
                        help="Path to annotation file like train.txt or val.txt")
    parser.add_argument('output_annotation', help="Path to output json file")
    parser.add_argument('--merge_all_person_categories', action='store_true',
                        help='To merge pedestrians, riders and partially-visible persons and '
                             'filter out other categories.')

    return parser.parse_args()


def parse_wider_gt(ann_file):
    with open(ann_file) as read_file:
        img_names = [line.strip() for line in read_file.readlines()]

    data_dir = os.path.dirname(ann_file)
    ann_paths = [os.path.join(data_dir, 'Annotations', image_name + '.jpg' + '.txt') for image_name in img_names]
    img_paths = [os.path.join(data_dir, 'Images', img_name + '.jpg') for img_name in img_names]
    boxes = defaultdict(list)
    for ann_path, img_path in zip(ann_paths, img_paths):
        with open(ann_path) as read_file:
            content = [line.strip() for line in read_file]
            num_boxes = int(content[0])
            for i in range(1, num_boxes + 1):
                label, xmin, ymin, xmax, ymax = [int(x) for x in content[i].split()]
                box = [xmin, ymin, xmax - xmin, ymax - ymin, label]
                boxes[img_path].append(box)
    return boxes


def convert_wider_annotation(ann_file, out_file, merge_all_person_categories):
    """ Converts wider annotation to COCO format. """

    img_id = 0
    ann_id = 0

    ann_dict = {}
    categories = [{"id": 1, "name": 'pedestrians', "supercategory": ''},
                  {"id": 2, "name": 'riders', "supercategory": ''},
                  {"id": 3, "name": 'partially-visible persons', "supercategory": ''},
                  {"id": 4, "name": 'ignore regions', "supercategory": ''},
                  {"id": 5, "name": 'crowd', "supercategory": ''}]

    images_info = []
    annotations = []

    boxes = parse_wider_gt(ann_file)

    for img_path, gt_bboxes in tqdm(boxes.items()):
        image_info = {}
        image_info['id'] = img_id
        img_id += 1
        image_info['width'], image_info['height'] = imagesize.get(img_path)
        image_info['file_name'] = os.path.relpath(img_path, os.path.dirname(out_file))
        images_info.append(image_info)

        for gt_bbox in gt_bboxes:
            if merge_all_person_categories:
                if gt_bbox[4] in [4, 5]:
                    continue
                gt_bbox[4] = 1
            ann = {
                'id': ann_id,
                'image_id': image_info['id'],
                'segmentation': [],
                'category_id': gt_bbox[4],
                'iscrowd': 0,
                'area': gt_bbox[2] * gt_bbox[3],
                'bbox': gt_bbox
            }
            ann_id += 1
            annotations.append(ann)

    ann_dict['images'] = images_info
    if merge_all_person_categories:
        ann_dict['categories'] = [{"id": 1, "name": 'person', "supercategory": ''}]
    else:
        ann_dict['categories'] = categories
    ann_dict['annotations'] = annotations
    os.makedirs(os.path.dirname(out_file), exist_ok=True)

    with open(out_file, 'w') as outfile:
        outfile.write(json.dumps(ann_dict))


def main():
    """ Main function. """

    args = parse_args()
    convert_wider_annotation(args.input_annotation, args.output_annotation, args.merge_all_person_categories)


if __name__ == '__main__':
    main()
