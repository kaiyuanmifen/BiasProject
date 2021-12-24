import os
import nltk
import pandas as pd
from pandas.io.parsers import ParserError
import tqdm
import ntpath
import argparse

BOS_WORD = '<s>'
#BOS_WORD = "<BOS>"
EOS_WORD = '</s>'
#EOS_WORD = "<EOS>"
PAD_WORD = '<pad>'
#PAD_WORD = "<PAD>"
UNK_WORD = '<unk>'
#UNK_WORD = "<UNK>"

def buid_dict_file(file_list, save_to, text_column, references_files = []):
    word_to_id = {}
    for file_item in file_list:
        try :
            df = pd.read_csv(file_item)
        except ParserError : # https://stackoverflow.com/questions/33998740/error-in-reading-a-csv-file-in-pandascparsererror-error-tokenizing-data-c-err
            df = pd.read_csv(file_item, lineterminator='\n')
        #for row in df.iterrows() : 
        for row in tqdm.tqdm(list(df.iterrows()), desc="%s" % file_item):
            text = row[1][text_column].strip()
            word_list = nltk.word_tokenize(text)
            for word in word_list:
                word = word.lower()
                word_to_id[word] = word_to_id.get(word, 0) + 1

    for file_item in references_files:
        try :
            df = pd.read_csv(file_item)
        except ParserError : # https://stackoverflow.com/questions/33998740/error-in-reading-a-csv-file-in-pandascparsererror-error-tokenizing-data-c-err
            df = pd.read_csv(file_item, lineterminator='\n')
        #for row in df.iterrows() : 
        for row in tqdm.tqdm(list(df.iterrows()), desc="%s" % file_item):
            text = row[1][text_column].strip()
            item1, item2 = text.split('\t')
            for item in [item1, item2]:
                word_list = nltk.word_tokenize(item)
                for word in word_list:
                    word = word.lower()
                    word_to_id[word] = word_to_id.get(word, 0) + 1

    print("Get word_dict success: %d words" % len(word_to_id))
    # write word_to_id to file
    word_dict_list = sorted(word_to_id.items(), key=lambda d: d[1], reverse=True)
    dict_file = os.path.join(save_to, "word_to_id.txt")
    with open(dict_file, 'w') as f:
        f.write("%s\n"%PAD_WORD)
        f.write("%s\n"%UNK_WORD)
        f.write("%s\n"%BOS_WORD)
        f.write("%s\n"%EOS_WORD)
        for ii in word_dict_list:
            f.write("%s\t%d\n" % (str(ii[0]), ii[1]))
            # f.write("%s\n" % str(ii[0]))
    print("build dict finished!") 
    return dict_file

# file_list, dict_file, data_columns
def load_word_dict(dict_file):
    word_dict = {}
    num = 0
    with open(dict_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            item = line.strip()
            word = item.split('\t')[0]
            word_dict[word] = num
            num += 1

    return word_dict

def path_leaf(path : str):
    """
    Returns the name of a file given its path
    https://stackoverflow.com/questions/8384737/extract-file-name-from-path-no-matter-what-the-os-path-format
    """
    head, tail = ntpath.split(path)
    return tail or ntpath.basename(head)

def get_csv_path(file_item, save_to):
    file_name = path_leaf(path = file_item)
    file_path = file_item.split(file_name)[0]
    file_name, extension = os.path.splitext(file_name) 
    file_name = file_name + "_" + extension.replace(".", "")
    csv_file = file_name+".csv"
    if os.path.isfile(csv_file):
        i = 1
        while os.path.isfile(file_name+'.'+str(i)+".csv"):
            i += 1
        csv_file = file_name+'.'+str(i)+'.csv'
    if save_to is not None :
        return  os.path.join(save_to, csv_file)
    else :
        return os.path.join(file_path, csv_file)

def build_id_file(file_list, dict_file, data_columns, references_files = [], save_to = None):

    assert len(data_columns) == 2
    text_column = data_columns[0]
    label_column = data_columns[1]

    # load word_dict
    word_dict = load_word_dict(dict_file)
    print("Load embedding success! Num: %d" % len(word_dict))

    # generate id file
    for file_item in file_list:
        try :
            df = pd.read_csv(file_item)
        except ParserError : # https://stackoverflow.com/questions/33998740/error-in-reading-a-csv-file-in-pandascparsererror-error-tokenizing-data-c-err
            df = pd.read_csv(file_item, lineterminator='\n')

        id_file_data = []
        #for row in df.iterrows() : 
        for row in tqdm.tqdm(list(df.iterrows()), desc="%s" % file_item):
            text = row[1][text_column].strip()
            word_list = nltk.word_tokenize(text)
            id_list = []
            for word in word_list:
                word = word.lower()
                id = word_dict[word]
                id_list.append(id)
            #id_file_data.append(id_list)
            id_file_data.append("%s\n" % (' '.join([str(k) for k in id_list])))

        csv_path = get_csv_path(file_item, save_to)
        pd.DataFrame(zip(id_file_data, list(df[label_column]))).to_csv(csv_path, header= [text_column, label_column])


    for file_item in references_files:
        try :
            df = pd.read_csv(file_item)
        except ParserError : # https://stackoverflow.com/questions/33998740/error-in-reading-a-csv-file-in-pandascparsererror-error-tokenizing-data-c-err
            df = pd.read_csv(file_item, lineterminator='\n')
        id_file_data = []
        #for row in df.iterrows() : 
        for row in tqdm.tqdm(list(df.iterrows()), desc="%s" % file_item):
            text = row[1][text_column].strip()
            item1, item2 = text.split('\t')
            # 1
            word_list1 = nltk.word_tokenize(item1)
            id_list1 = []
            for word in word_list1:
                word = word.lower()
                id = word_dict[word]
                id_list1.append(id)
            # 2
            word_list2 = nltk.word_tokenize(item2)
            id_list2 = []
            for word in word_list2:
                word = word.lower()
                id = word_dict[word]
                id_list2.append(id)
            
            id_file_data.append([id_list1, id_list2])
            id_file_data.append("%s\t%s\n" % (' '.join([str(k) for k in id_list1]), ' '.join([str(k) for k in id_list2])))
        
        csv_path = get_csv_path(file_item, save_to)
        pd.DataFrame(zip(id_file_data, list(df[label_column]))).to_csv(csv_path, header= [text_column, label_column])

    print('build id file finished!')

if __name__ == '__main__':
    # parse parameters
    parser = argparse.ArgumentParser(description="preprocessed_data")

    # main parameters
    parser.add_argument("-f", "--file_list", type=str, help="file1,file2, ...")
    parser.add_argument("-rf", "--references_files", type=str, default="", help="ref_file1,ref_file2, ...")
    parser.add_argument("-dc", "--data_columns", type=str, default="c1,c2,..", help="")
    parser.add_argument("-st", "--save_to", type=str, default="", help="")

    args = parser.parse_args()
    file_list = args.file_list.strip().split(",")
    assert all([os.path.isfile(f) for f in file_list])
    references_files = args.references_files.strip().strip('"')
    if references_files != "" :
        references_files = references_files.split(",")
        assert all([os.path.isfile(f) for f in references_files])
    else :
        references_files = []
    
    save_to = args.save_to
    if not os.path.exists(save_to) :
        os.mkdir(save_to)

    data_columns = args.data_columns
    data_columns = data_columns.split(",")
    assert len(data_columns) == 2
    text_column = data_columns[0]
    #label_column = data_columns[1]

    dict_file = buid_dict_file(file_list, save_to, text_column, references_files = references_files)
    build_id_file(file_list, dict_file, data_columns, references_files = references_files, save_to = save_to)