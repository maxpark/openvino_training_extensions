import argparse
import os
from subprocess import run
import tempfile

import cv2
import mmcv
import numpy as np

from mmcv import DictAction
from mmdet.datasets import build_dataset
from pycocotools.mask import decode
from tqdm import tqdm


def convert_to_wider(config, input, out_folder, update_config):
    """ Main function. """

    if input is not None and not input.endswith(('.pkl', '.pickle')):
        raise ValueError('The input file must be a pkl file.')

    cfg = mmcv.Config.fromfile(config)
    if update_config:
        cfg.merge_from_dict(update_config)
    dataset = build_dataset(cfg.data.test)

    results = mmcv.load(input)

    totaltext_friendly_results = []
    for i, sample in enumerate(tqdm(dataset)):
        filename = sample['img_metas'][0].data['filename']
        folder, image_name = filename.split('/')[-2:]
        totaltext_friendly_results.append({'folder': folder, 'name':  'res_' + image_name[:-4] + '.txt',
                                           'boxes': results[i][0][0],
                                           'segms': results[i][1][0], 'texts': results[i][2]})

    for result in totaltext_friendly_results:
        folder = os.path.join(out_folder, result['folder'])
        os.makedirs(folder, exist_ok=True)
        with open(os.path.join(folder, result['name']), 'w') as write_file:

            for box, segm, text in zip(result['boxes'], result['segms'], result['texts']):
                text = text.upper()
                mask = decode(segm)
                contours = cv2.findContours(
                    mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
                if contours:
                    contour = sorted(
                        contours, key=lambda x: -cv2.contourArea(x))[0]
                    contour = cv2.boxPoints(cv2.minAreaRect(contour)).reshape(-1)
                else:
                    print('Used bbox')
                    xmin, ymin, xmax, ymax, conf = box
                    contour = [xmin, ymin, xmax, ymin, xmax, ymax, xmin, ymax]
                
                res_str = ','.join([str(int(round(x))) for x in contour]) + f',{text}'

                write_file.write(res_str + '\n')
                # mask = np.zeros(mask.shape, dtype=np.uint8)
                # mask = cv2.drawContours(mask, [contour], -1, 255, -1)
                # write_file.write(res_str + '\n')
                # cv2.imshow('mask', mask)
                # cv2.waitKey(0)


def parse_args():
    """ Parses input arguments. """

    parser = argparse.ArgumentParser(
        description='This script converts output of test.py (mmdetection) to '
                    'a set of files that can be passed to official WiderFace '
                    'evaluation procedure.')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('snapshot', help='model weights')
    parser.add_argument('--icdar15_ann', default='/media/ikrylov/datasets/text_spotting2/icdar2015_test.json')
    parser.add_argument('--icdar15_root', default='/media/ikrylov/datasets/text_spotting2')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    mmdetection_test_py = '../../external/mmdetection/tools/test.py'
    temp_dir = tempfile.mkdtemp()
    out_file = os.path.join(temp_dir, 'res.pkl')
    icdar15_output = os.path.join(temp_dir, 'icdar15')

    run(f'python {mmdetection_test_py}'
        f' {args.config}'
        f' {args.snapshot}'
        f' --out {out_file}'
        f' --update_config data.test.ann_file={args.icdar15_ann} data.test.img_prefix={args.icdar15_root}', shell=True)


    convert_to_wider(args.config, out_file, icdar15_output,
                     {'data.test.ann_file': args.icdar15_ann, 'data.test.img_prefix': args.icdar15_root}
                    )
    
    run(f'cd {icdar15_output}/ch4_test_images/; zip Test.zip *', shell=True)


    run(f'cd ../../../MaskTextSpotterV3/evaluation/icdar2015/e2e;'
        f'python script.py --s {icdar15_output}/ch4_test_images/Test.zip', 
    shell=True)
