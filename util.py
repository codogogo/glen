import codecs
import os
from scipy.stats import spearmanr
from scipy.stats import pearsonr

def get_directory_files(dir_path):
    files = [f for f in list(os.walk(dir_path))[0][2]]
    return (files, [os.path.join(dir_path, f) for f in files])

def load_lines(path):
	return [l.strip() for l in list(codecs.open(path, "r", encoding = 'utf8', errors = 'replace').readlines())]

def write_lines(path, list, append = False):
	f = codecs.open(path,"a" if append else "w",encoding='utf8')
	for l in list:
		f.write(str(l) + "\n")
	f.close()

def write_text(path, text, append = False):
    f = codecs.open(path,"a" if append else "w",encoding='utf8')
    f.write(text + "\n")
    f.close()

def load_csv_lines(path, delimiter = ',', indices = None):
	f = codecs.open(path,'r',encoding='utf8', errors = 'ignore')
	lines = [l.strip().split(delimiter) for l in f.readlines()]
	if indices is None:
		return lines
	else:
		return [sublist(l, indices) for l in lines if len(l) >= max(indices) + 1]

def sublist(list, indices):
	sublist = []
	for i in indices:	
		sublist.append(list[i])
	return sublist
