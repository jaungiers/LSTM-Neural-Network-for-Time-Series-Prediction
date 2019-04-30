import unittest
from data_processor import DataLoader

class testDataLoader(unittest.TestCase):
    """
    Test class used for test on external APIs of DataLoader

    This is useful when you want to replace the DataLoader 
    with a custom module that works with arbitrary data.
    All public methods are tested beside normalise_windows 
    that is not involved in data flow manipulation

    NOTE: All the tests are to be run with ../data/sp500.csv 
    from the original github repo
    """

    def setUp(self):
        """
        I will write all the tests for the data flow
        and then just modify the setup in order to handle
        two-sigma specs dataset instead of reading csv
        """
        self.dl = DataLoader("../data/sp500.csv",0.8,["Close"])
        #import pdb;pdb.set_trace()

    def test_data(self):
        # if this breaks means data is not in the right path
        # or is not formatted in the correct way
        self.assertTrue(isinstance(self.dl, DataLoader))
        print("[testDataLoader] test_data OK")

    def test_get_test_data(self):
        x, y = self.dl.get_test_data(10, 0)
        self.assertEqual(y.shape, (930, 1))
        self.assertEqual(x.shape, (930, 9, 1))

        for j in range(len(x) - len(x[0]) - 1):
            for i, _ in enumerate(x[j]):
                # assert that lagged values are correct
                self.assertEqual(x[j][i][0], x[j+i][0][0])

        print("[testDataLoader] test_get_test_data OK")

    def test_get_train_data(self):
        x, y = self.dl.get_train_data(10, 0)
        self.assertEqual(y.shape, (3747, 1))
        self.assertEqual(x.shape, (3747, 9, 1))

        for j in range(len(x) - len(x[0]) - 1):
            for i, _ in enumerate(x[j]):
                # assert that lagged values are correct
                self.assertEqual(x[j][i][0], x[j+i][0][0])

        print("[testDataLoader] test_get_train_data OK")

    def test_generate_train_batch(self):
        generator = self.dl.generate_train_batch(10, 20, 0)
        x, y = next(generator)
        self.assertEqual(y.shape, (20, 1))
        self.assertEqual(x.shape, (20, 9, 1))

        for j in range(len(x) - len(x[0]) - 1):
            for i, _ in enumerate(x[j]):
                # assert that lagged values are correct
                self.assertEqual(x[j][i][0], x[j+i][0][0])

        print("[testDataLoader] test_generate_train_batch OK")

if __name__ == "__main__":
    unittest.main()
