import fasttext
import os


from language_identification.prepare_data import select_data


class LanguageIdentification():
    def __init__(self) -> None:
        model_directory = 'model/'
        self.model_path = model_directory + 'language_identification_model.bin'

        if not os.path.isfile(self.model_path):
            print('Language Detection model not present')
            if not os.path.exists(model_directory):
                os.makedirs(model_directory)
            self.train_model()

        self.model = fasttext.load_model(self.model_path)
        print('Model loaded from ', self.model_path)

    def train_model(self):
        train_file_path = 'data/train.txt'

        if not os.path.isfile(train_file_path):
            print('Train file not present')
            select_data()

        print('Model training starting')
        model = fasttext.train_supervised(
            train_file_path, dim=16, minn=2, maxn=4, loss='hs')

        model.save_model(self.model_path)

        print('Model trained and saved to ', self.model_path)

        self.print_results(*model.test('data/test.txt'))

    def predict(self, text):
        prediction = self.model.predict(text)
        lang = prediction[0][0].replace('__label__', '')
        score = prediction[1][0]
        return {'language_code': lang, 'score': score}

    def print_results(self, N, p, r):
        print('Evaluation of the model on the test set\n')
        print("N\t" + str(N))
        print("P@{}\t{:.3f}".format(1, p))
        print("R@{}\t{:.3f}".format(1, r))

    def test_model(self):
        self.print_results(*self.model.test('data/test.txt'))


if __name__ == '__main__':
    language_identification = LanguageIdentification()

    print(language_identification.predict('this is a test for english'))
