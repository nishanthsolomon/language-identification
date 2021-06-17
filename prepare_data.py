import wget
import bz2
import shutil
import tarfile
import os
import pandas as pd


data_directory = './data/'


def download_data():

    download_file = data_directory + 'sentences.tar.bz2'
    tar_file = data_directory + 'sentences.tar'

    url = 'http://downloads.tatoeba.org/exports/sentences.tar.bz2'
    print('Downloading file from ', url)
    wget.download(url, out=data_directory)
    print('Downloaded file')

    print('Extracting file')
    with bz2.BZ2File(download_file) as fr, open(tar_file, "wb") as fw:
        shutil.copyfileobj(fr, fw)

    tarfile.open(tar_file).extractall(data_directory)

    os.remove(download_file)
    os.remove(tar_file)
    print('Extracted file')


def select_languages(languages):
    languages = languages.groupby(
        ['language']).size().reset_index(name='count')
    languages = languages[languages['count'] > 100]

    return languages


def select_data():
    sentences_file = data_directory + 'sentences.csv'

    if not os.path.isfile(sentences_file):
        print('Sentences file not present')
        download_data()

    print('Selecting languages where the number of sentences present is greater than 100')
    sentences_df = pd.read_csv(
        sentences_file, sep='\t', names=['language', 'text'])

    sentences_df['language'] = '__label__' + \
        sentences_df['language'].astype(str)

    selected_languages = select_languages(sentences_df[['language']])

    train_dfs = []
    test_dfs = []

    print('Splitting the selected sentences into train and test set')
    for _, row in selected_languages.iterrows():
        select_df = sentences_df[sentences_df['language'] == row['language']]
        n = len(select_df)//10

        test_dfs.append(select_df.iloc[:n])
        train_dfs.append(select_df.iloc[n:])

    train_df = pd.concat(train_dfs).sample(frac=1)
    test_df = pd.concat(test_dfs).sample(frac=1)

    train_df.to_csv('data/train.txt', sep=' ', header=False, index=False)
    test_df.to_csv('data/test.txt', sep=' ', header=False, index=False)

    print('Train and test files created')


if __name__ == '__main__':
    select_data()
