import os
import json
import pandas as pd
from glob import glob
from tqdm import tqdm

def main():

    image_data_dir = '/workspace/data/image'
    grp_list = ['p10', 'p11', 'p12', 'p13', 'p14', 'p15', 'p16', 'p17' , 'p18', 'p19']
    if not os.path.exists('./image_text_pair'):
        os.makedirs('./image_text_pair')

    for _grp in grp_list:
        output_json_file = os.path.join('./image_text_pair', _grp + '.json')

        grp_dir = os.path.join(image_data_dir, _grp)
        json_file_name = os.path.join('./processed_text/data_list_' + _grp + '.json')
        print(grp_dir)
        image_list = glob(grp_dir + '/**/*.jpg', recursive=True)
        new_image_list = [] # containts only images
        for _image in image_list:
            if "mask" not in _image:
                new_image_list.append(_image)

        text_dict = json.load(open(json_file_name, 'r'))

        out_text_image_dict = {}
        for _image in tqdm(new_image_list):
            image_name = _image.split('/')

            key = _grp + '_' + image_name[-3] + '_' + image_name[-2]


            if key in text_dict.keys():
                subject_text_dict = text_dict[key]
                if subject_text_dict is not None:
                    if 'FINDINGS' in subject_text_dict.keys():
                        if subject_text_dict['FINDINGS'] is not None and subject_text_dict['FINDINGS'] != '':
                            out_text_image_dict[key] = {'FINDINGS': subject_text_dict['FINDINGS'], 'IMAGE': _image}
            else:
                pass


        with open(output_json_file, 'w') as f:
            json.dump(out_text_image_dict, f, indent = 4)



if __name__ == '__main__':
    main()