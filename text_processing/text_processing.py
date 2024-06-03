import json
import random

import pandas as pd
import os
from glob import glob
import re
import random
from random import shuffle
from tqdm import tqdm


def generate_filelist(filename):
    fid = open(filename, 'r')
    data = []
    count = 0
    while True:
        count += 1
        line = fid.readline()
        data.append(line.rstrip())
        if not line:
            break
    print('Reading complete, Number of lines: {}'.format(count))
    return data


def write_filelist(data_dir, out_filename):
    file_list = glob(data_dir + '/**/*.txt', recursive=True)
    with open(out_filename, 'w') as f:
        for _line in file_list:
            f.write(_line + '\n')
    f.close()


def extract_sentences(line):
    res = re.findall(r'\w+', line)
    return res


def create_sentences(word_list, flag):
    sentences = ""
    if flag == 1:
        for _ in word_list[1:]:
            sentences = sentences + _ + ' '
        return sentences
    else:
        for _ in word_list:
            sentences = sentences + _ + ' '
        return sentences


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


def check_word_mixed_case(word):
    if word[0].isupper() and word[1:].islower():
        return True
    elif word.islower():
        return True
    else:
        return False


def find_factors_number(n):
    factors = []
    for i in range(1, n + 1):
        if n % i == 0:
            factors.append(i)
    return factors


def remove_empty_string(word_list):
    res = []
    for _ in word_list:
        if _ != '':
            res.append(_)
    return res

def divide_list(lines, n):
    line_list = []
    for i in range(n):
        line_list.append([])
    for i in range(len(lines)):
        line_list[i % n].append(lines[i])
    return line_list

def main():
    grp = 'p11'
    data_dir = '/workspace/data/mimic-cxr-reports/files' + '/' + grp
    filename = './rad_reports_' + grp + '.txt'
    # write_filelist(data_dir, filename)
    with open(filename, 'r') as f:
        lines = f.readlines()
    line_list = divide_list(lines, 10)
    out_dir_base_name = '/workspace/app/data_list_' + grp + '_'
    for i in tqdm(range(len(line_list))):
        out_filename = out_dir_base_name + str(i) + '.json'
        print(out_filename)
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

def check_keys(line_list, keys):
    pass

def create_findings_impression_dict(fn):
    line_list = parse_file(fn.rstrip())
    clean_lines = []
    for _line in line_list:
        a = remove_empty_string(_line.split(':'))
        _clean_lines = []
        for _a in a:
            _clean_lines.append(_a.lstrip())
        clean_lines.append(_clean_lines)

    #Check if finding or impression exist
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




    """
    data_dict = {}

    if findings_idx is None:
        data_dict['FINDINGS'] = None
    else:
        if impression_idx > findings_idx:
            findings_text = range(findings_idx + 1, impression_idx)
            if len(clean_lines[findings_idx]) > 1:
                findings = clean_lines[findings_idx][1]
            else:
                findings = ""
            for i in findings_text:
                findings = findings + ' ' + clean_lines[i][0]

        data_dict['FINDINGS'] = findings

    if impression_idx is None:
        data_dict['IMPRESSION'] = None
    else:
        if findings_idx > impression_idx:
            impression_text = range(impression_idx + 1, findings_idx)
            if len(clean_lines[impression_idx]) > 1:
                impression = clean_lines[impression_idx][1]
            else:
                impression = ""
            for i in impression_text:
                impression = impression + ' ' + clean_lines[i][0]
        data_dict['IMPRESSION'] = impression

    print(data_dict)


    """

    keys = ['INDICATION', 'TECHNIQUE', 'COMPARISON', 'FINDINGS', 'IMPRESSION']
    '''
    keys = ['INDICATION', 'TECHNIQUE', 'COMPARISON', 'FINDINGS', 'IMPRESSION']
    key_index_list = []
    print(line_list)

    for _line in line_list:
        if 'FINDINGS' in _line or 'Findings' in _line or 'FINDING' in _line or 'Finding' in _line:
            findings_idx = line_list.index(_line)
        else:
            findings_idx = None
    for _line in line_list:
        if 'IMPRESSION' in _line or 'Impression' in _line or 'IMPRESSIONS' in _line or 'Impressions' in _line:
            impression_idx = line_list.index(_line)
        else:
            impression_idx = None

    print(findings_idx, impression_idx)
    # extract findings
    findings_text = range(findings_idx + 1, impression_idx)
    # extract impression
    impression_text = range(impression_idx + 1, len(clean_lines))

    if len(clean_lines[findings_idx]) > 1:
        findings = clean_lines[findings_idx][1]
    else:
        findings = ""
    for i in findings_text:
        findings = findings + ' ' + clean_lines[i][0]
    if len(clean_lines[impression_idx]) > 1:
        impression = clean_lines[impression_idx][1]
    else:
        impression = ""
    for i in impression_text:
        impression = impression + ' ' + clean_lines[i][0]

    data_dict = {'FINDINGS': findings, 'IMPRESSION': impression}
    print(data_dict)
    '''


if __name__ == '__main__':
    main()
