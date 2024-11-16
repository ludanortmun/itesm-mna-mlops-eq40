import unittest
import random

import numpy as np
import pandas as pd

from mlops.split import load, split

# This is relative to the root of the repository
sample_dataset_path = 'mlops/tests/fixtures/sample_dataset.csv'
sample_dataset_len = 5


class SplitTest(unittest.TestCase):
    def test_load(self):
        x, y = load(sample_dataset_path)
        self.assertIsInstance(x, pd.DataFrame)
        self.assertIsInstance(y, pd.Series)

    # The split function is not actually concerned with the actual data,
    # so we can use the dataset directly to test the function, without column transformation
    def test_split_test_set_is_20_percent(self):
        x = []
        y = []
        for i in range(100):
            x.append(np.arange(10))
            y.append(random.choice([0, 1]))

        # First assert that the mock dataset is correctly generated
        # There should be 100 samples, each with 10 features
        self.assertEqual(len(x), 100)
        self.assertEqual(len(x[0]), 10)
        self.assertEqual(len(y), 100)

        x_train, x_test, y_train, y_test = split(x, y)

        self.assertEqual(len(x_train), 80)
        self.assertEqual(len(x_test), 20)
        self.assertEqual(len(y_train), 80)
        self.assertEqual(len(y_test), 20)


    def test_load_with_invalid_path(self):
        with self.assertRaises(FileNotFoundError):
            load('invalid_path.csv')



if __name__ == '__main__':
    unittest.main()
