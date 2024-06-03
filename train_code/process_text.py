import json 
import random 
import pandas as pd
from glob import glob
import os 
import re
from random import shuffle
from tqdm import tqdm

def parse_file(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()

    keys = ['EXAMINATION', 'INDICATION', 'TECHNIQUE', 'COMPARISON', 'FINDINGS', 'IMPRESSION']
    data_dict = {}

    # Remove the empty lines
    line_list = []
    for i in range(len(lines)):
        res = extract_sentences(lines[i])
        if len(res) > 1:
            line_list.append(lines[i].rstrip())
        elif len(res) == 1:
            line_list.append(lines[i].rstrip())
    #####
    return line_list

def extract_sentences(line):
    res = re.findall(r'\w+', line)
    return res


def remove_empty_string(word_list):
    res = []
    for _ in word_list:
        if _ != '':
            res.append(_)
    return res

def divide_list(lines, n):
    """
    divide the list into n parts
    """
    line_list = []
    for i in range(n):
        line_list.append([])
    for i in range(len(lines)):
        line_list[i % n].append(lines[i])
    return line_list

def create_findings_impression_dict(fn):
    line_list = parse_file(fn.rstrip())
    clean_lines = []
    for _line in line_list:
        a = remove_empty_string(_line.split(':'))
        _clean_lines = []
        for _a in a:
            _clean_lines.append(_a.lstrip())
        clean_lines.append(_clean_lines)

    # Check if finding or impression exist
    for _line in clean_lines:
        if 'FINDINGS' in _line or 'Findings' in _line or 'FINDING' in _line or 'Finding' in _line:
            findings_idx = clean_lines.index(_line)
            break
        else:
            findings_idx = None

    for _line in clean_lines:
        if 'IMPRESSION' in _line or 'Impression' in _line or 'IMPRESSIONS' in _line or 'Impressions' in _line:
            impression_idx = clean_lines.index(_line)
            break
        else:
            impression_idx = None
    """
    4 cases
    only findings
    only impression
    findings > impression
    impression > findings


    """
    # Case 1: Only findings
    if findings_idx is not None and impression_idx is None:
        findings_text = range(findings_idx + 1, len(clean_lines))
        if len(clean_lines[findings_idx]) > 1:
            findings = clean_lines[findings_idx][1]
        else:
            findings = ""
        for i in findings_text:
            findings = findings + ' ' + clean_lines[i][0]
        data_dict = {}
        data_dict['FINDINGS'] = findings
        data_dict['IMPRESSION'] = None
        return data_dict

    # Case 2: Only impression
    if impression_idx is not None and findings_idx is None:
        impression_text = range(impression_idx + 1, len(clean_lines))
        if len(clean_lines[impression_idx]) > 1:
            impression = clean_lines[impression_idx][1]
        else:
            impression = ""
        for i in impression_text:
            impression = impression + ' ' + clean_lines[i][0]
        data_dict = {}
        data_dict['FINDINGS'] = None
        data_dict['IMPRESSION'] = impression
        return data_dict

    # Case 3: impression > findings
    if findings_idx is not None and impression_idx is not None:
        if impression_idx > findings_idx:
            findings_text = range(findings_idx + 1, impression_idx)
            if len(clean_lines[findings_idx]) > 1:
                findings = clean_lines[findings_idx][1]
            else:
                findings = ""
            for i in findings_text:
                findings = findings + ' ' + clean_lines[i][0]
            impression_text = range(impression_idx + 1, len(clean_lines))
            if len(clean_lines[impression_idx]) > 1:
                impression = clean_lines[impression_idx][1]
            else:
                impression = ""
            for i in impression_text:
                impression = impression + ' ' + clean_lines[i][0]
            data_dict = {}
            data_dict['FINDINGS'] = findings
            data_dict['IMPRESSION'] = impression
            return data_dict

        # Case 4: f > findings
        if findings_idx > impression_idx:
            print('case 4')
            impression_text = range(impression_idx + 1, findings_idx)
            print(impression_text)
            if len(clean_lines[impression_idx]) > 1:
                impression = clean_lines[impression_idx][1]
            else:
                impression = ""
            for i in impression_text:
                impression = impression + ' ' + clean_lines[i][0]
            findings_text = range(findings_idx + 1, len(clean_lines))
            if len(clean_lines[findings_idx]) > 1:
                findings = clean_lines[findings_idx][1]
            else:
                findings = ""
            for i in findings_text:
                findings = findings + ' ' + clean_lines[i][0]
            data_dict = {}
            data_dict['FINDINGS'] = findings
            data_dict['IMPRESSION'] = impression
            return data_dict

def combine_json_files(json_files, out_filename):
    data_dict = {}
    for _file in json_files:
        with open(_file, 'r') as f:
            _data_dict = json.load(f)
        data_dict.update(_data_dict)
    with open(out_filename, 'w') as f:
        json.dump(data_dict, f, indent = 4)

def create_regular_text_file(fn, output_filename):
    """
    Create regular text file by extracting the findings from the report
    """
    fid = open(fn, 'r')
    data_dict = json.load(fid)
    fid.close()
    a = []
    for key in data_dict.keys():
        if data_dict[key] is not None:
            if 'FINDINGS' in data_dict[key].keys():
                if data_dict[key]['FINDINGS'] is not None:
                    a.append(data_dict[key]['FINDINGS'])

            if 'IMPRESSION' in data_dict[key].keys():
                if data_dict[key]['IMPRESSION'] is not None:
                    a.append(data_dict[key]['IMPRESSION'])

    # write a list to a file
    fp = open(output_filename, 'w')
    for item in a:
        fp.write("%s\n" % item)
    fp.close()





def main():
    grp_list = ['p10', 'p11', 'p12', 'p13', 'p14', 'p15', 'p16', 'p17' , 'p18', 'p19']
    out_dir = '/workspace/app/processed_text'
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    for grp in grp_list:
        text_dir = '/workspace/data/text' + '/' + grp

        filename_list = glob(text_dir + '/**/*.txt', recursive=True)
        line_list = divide_list(filename_list, 10)
        out_dir_base_name = out_dir + '/data_list_' + grp + '_'
        json_file_list = []
        for i in tqdm(range(len(line_list))):
            out_filename = out_dir_base_name + str(i) + '.json'
            data_dict = {}
            for _line in tqdm(line_list[i]):
                _word_list = _line.rstrip().split('/')
                key = _word_list[-3] + '_' + _word_list[-2] + '_' + _word_list[-1].split('.')[0]
                _word_list = _line.rstrip().split('/')
                key = _word_list[-3] + '_' + _word_list[-2] + '_' + _word_list[-1].split('.')[0]
                data_dict[key] = create_findings_impression_dict(_line.rstrip())
            fid = open(out_filename, 'w')
            json.dump(data_dict, fid, indent=4)
            fid.close()
            json_file_list.append(out_filename)
        out_filename = out_dir + '/data_list_' + grp + '.json'
        combine_json_files(json_file_list, out_filename)


if __name__ == '__main__':
    grp_list = ['p10', 'p11', 'p12', 'p13', 'p14', 'p15', 'p16', 'p17' , 'p18', 'p19']
    for _grp in grp_list:
        output_file = './processed_text/data_list_' + _grp + '.txt'
        create_regular_text_file('./processed_text/data_list_' + _grp + '.json', output_file)
