import wget
import bz2
import shutil
import tarfile
import os
import pandas as pd

'''
1. data/train.txt, data/test.txt
2. sentences.csv
3. sentences.tar.bz2
4. data directory
'''


class PrepareData():
    def __init__(self, data_directory='data/') -> None:
        self.data_directory = data_directory

        self.sentences_tar_bz_file = data_directory + 'sentences.tar.bz2'
        self.sentences_tar_file = data_directory + 'sentences.tar'
        self.sentences_file = data_directory + 'sentences.csv'

        self.train_file = data_directory + 'train.txt'
        self.test_file = data_directory + 'test.txt'

    def check_files(self):
        if os.path.isfile(self.train_file) and os.path.isfile(self.test_file):
            print('Train and test files found')
        elif os.path.isfile(self.sentences_file):
            self.select_data()
        elif os.path.isfile(self.sentences_tar_bz_file):
            self.extract_file()
            self.select_data()
        elif os.path.exists(self.data_directory):
            self.download_data()
            self.extract_file()
            self.select_data()
        else:
            os.makedirs(self.data_directory)
            self.download_data()
            self.extract_file()
            self.select_data()

        self.delete_files()

        return True

    def delete_files(self):
        try:
            os.remove(self.sentences_tar_bz_file)
            os.remove(self.sentences_tar_file)
            os.remove(self.sentences_file)
        except OSError:
            pass

    def download_data(self):
        url = 'http://downloads.tatoeba.org/exports/sentences.tar.bz2'
        print('Downloading file from ', url)
        wget.download(url, out=self.data_directory)
        print('Downloaded file')

    def extract_file(self):

        print('Extracting file')
        with bz2.BZ2File(self.sentences_tar_bz_file) as fr, open(self.sentences_tar_file, "wb") as fw:
            shutil.copyfileobj(fr, fw)

        tarfile.open(self.sentences_tar_file).extractall(self.data_directory)

        print('Extracted file')

    def select_languages(self, languages):
        languages = languages.groupby(
            ['language']).size().reset_index(name='count')
        languages = languages[languages['count'] > 100]

        return languages

    def select_data(self):
        print(
            'Selecting languages where the number of sentences present is greater than 100')
        sentences_df = pd.read_csv(
            self.sentences_file, sep='\t', names=['language', 'text'])

        sentences_df['language'] = '__label__' + \
            sentences_df['language'].astype(str)

        selected_languages = self.select_languages(sentences_df[['language']])

        train_dfs = []
        test_dfs = []

        print('Splitting the selected sentences into train and test set')
        for _, row in selected_languages.iterrows():
            select_df = sentences_df[sentences_df['language'].astype(
                str) == row['language']]
            n = len(select_df)//10

            test_dfs.append(select_df.iloc[:n])
            train_dfs.append(select_df.iloc[n:])

        train_df = pd.concat(train_dfs).sample(frac=1)
        test_df = pd.concat(test_dfs).sample(frac=1)

        train_df.to_csv('data/train.txt', sep=' ', header=False, index=False)
        test_df.to_csv('data/test.txt', sep=' ', header=False, index=False)

        print('Train and test files created')


# if __name__ == '__main__':
#     prepare_data = PrepareData()
#     prepare_data.check_files()
