import random
import unittest

import numpy as np
import pandas as pd
from mlops.preprocess import load, preprocess, split, run_preprocessing_pipeline

# This is relative to the root of the repository
sample_dataset_path = 'mlops/tests/fixtures/sample_dataset.csv'
sample_dataset_len = 5

class PreprocessTest(unittest.TestCase):

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

    def test_preprocess_output_shape(self):
        # 7 numeric columns + (2*5) binary columns one-hot encoded
        expected_cols = 17
        x = pd.read_csv(sample_dataset_path)
        x, _ = preprocess(x, x) # For the sake of this test, we will use the same dataset for both train and test
        self.assertEqual(x.shape, (sample_dataset_len, expected_cols))

    def test_preprocess_binary_cols_are_one_hot_encoded(self):
        # These are the expected indexes of the binary columns in the processed dataset
        # The first index is for the column mapping to 0, and the second index is for the column mapping to 1
        binary_col_indexes = {
            'anaemia': [7,8],
            'diabetes': [9,10],
            'high_blood_pressure': [11,12],
            'sex': [13,14],
            'smoking': [15,16]
        }
        x = pd.read_csv(sample_dataset_path)
        x_transformed, _ = preprocess(x, x) # For the sake of this test, we will use the same dataset for both train and test

        for i in range(sample_dataset_len):
            for col, indexes in binary_col_indexes.items():
                # The original value determines whether the first or the second one-hot encoded column should be 1
                original_value = x.loc[i, col]

                # If the original value is 0, the first index for the given column should be 1,
                # otherwise, when the original value is 1, the second index should be 1.
                self.assertEqual(x_transformed[i][indexes[original_value]], 1)

                # Now that we now that the correct column is 1, we can ensure that the other column is 0
                # by evaluating the sum of both columns
                self.assertEqual(x_transformed[i][indexes[0]] + x_transformed[i][indexes[1]], 1)

    def test_preprocess_numeric_cols_are_normalized(self):
        # These are the expected indexes of the numeric columns in the processed dataset
        numeric_col_indexes = [0,1,2,3,4,5,6]
        x = pd.read_csv(sample_dataset_path)
        x_transformed, _ = preprocess(x, x) # For the sake of this test, we will use the same dataset for both train and test

        for i in range(sample_dataset_len):
            for col in numeric_col_indexes:
                # The mean of the column should be 0
                self.assertAlmostEqual(np.mean(x_transformed[:, col]), 0, places=5)
                # The standard deviation of the column should be 1
                self.assertAlmostEqual(np.std(x_transformed[:, col]), 1, places=5)


    # This function checks that all steps in preprocessing (load, transform and split) are working correctly in tandem.
    def test_run_preprocessing_pipeline(self):
        expected_cols = 17
        expected_train_rows = 4
        expected_test_rows = 1

        x_train, x_test, y_train, y_test = run_preprocessing_pipeline(sample_dataset_path)

        # These assertions validate that the data is split correctly and the columns are transformed
        self.assertEqual(x_train.shape, (expected_train_rows, expected_cols))
        self.assertEqual(x_test.shape, (expected_test_rows, expected_cols))

        self.assertEqual(len(y_train), expected_train_rows)
        self.assertEqual(len(y_test), expected_test_rows)

    def test_run_preprocessing_pipeline_with_invalid_path(self):
        with self.assertRaises(FileNotFoundError):
            run_preprocessing_pipeline('invalid_path.csv')
    

if __name__ == '__main__':
    unittest.main()
