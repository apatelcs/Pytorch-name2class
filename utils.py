from __future__ import unicode_literals
import unicodedata
import string
import torch
import os
import glob

all_letters = string.ascii_letters + " .,;'"
num_letters = len(all_letters)

def unicode_to_ascii(s):
    out = ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn' and c in all_letters)
    return out

def read_lines(filename):
    lines = open(filename, encoding='utf-8').read().strip().split('\n')
    return [unicode_to_ascii(line) for line in lines]

def find_files(path):
    return glob.glob(path)

# list of categories
all_categories = []
# Dictionary category: list of names
category_lines = {}
num_names = 0

for filename in find_files('data/names/*.txt'):
    category = os.path.splitext(os.path.basename(filename))[0]
    all_categories.append(category)

    lines = read_lines(filename)
    num_names += len(lines)
    category_lines[category] = lines

num_categories = len(all_categories)

# print(f'Number of names: {num_names}')
# print(f'Number of categories: {num_categories}')

def letter_to_ind(letter):
    return all_letters.find(letter)

def line_to_tensor(line):
    # Inits tensor shape etters x 1 x num total letters
    tensor = torch.zeros(len(line), 1, num_letters)
    # One hot encoding for each letter in name
    for i, letter in enumerate(line):
        tensor[i][0][letter_to_ind(letter)] = 1
    return tensor

def load_all():
    lst_of_categories = []
    lst_of_lines = []
    lst_of_category_tensors = []
    lst_of_line_tensors = []
    for category in all_categories:
        for line in category_lines[category]:
            category_tensor = torch.tensor([all_categories.index(category)], dtype=torch.long)
            line_tensor = line_to_tensor(line)
            lst_of_category_tensors.append(category_tensor)
            lst_of_line_tensors.append(line_tensor)
            lst_of_categories.append(category)
            lst_of_lines.append(line)

    return lst_of_line_tensors, lst_of_category_tensors, lst_of_lines, lst_of_categories