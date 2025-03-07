import unittest
import os
import pandas as pd
from reaction_utils import read_reaction_file, validate_reaction_file

class TestReactionUtils(unittest.TestCase):
    def setUp(self):
        # Create a temporary test file
        self.test_file = "test_reactions.csv"
        self.test_data = pd.DataFrame({
            'smarts': ['[C:1][C:2]>>[C:1]C[C:2]', '[C:1]O>>[C:1]C'],
            'name': ['Alkene Addition', 'Alcohol Alkylation']
        })
        self.test_data.to_csv(self.test_file, header=False, index=False, sep='\t')

    def tearDown(self):
        # Clean up the test file
        if os.path.exists(self.test_file):
            os.remove(self.test_file)

    def test_read_reaction_file(self):
        smarts_list, names_list = read_reaction_file(self.test_file)
        self.assertEqual(len(smarts_list), 2)
        self.assertEqual(len(names_list), 2)
        self.assertEqual(smarts_list[0], '[C:1][C:2]>>[C:1]C[C:2]')
        self.assertEqual(names_list[0], 'Alkene Addition')

    def test_validate_reaction_file(self):
        self.assertTrue(validate_reaction_file(self.test_file))
        
        # Test with empty file
        empty_file = "empty.csv"
        pd.DataFrame(columns=['smarts', 'name']).to_csv(empty_file, index=False, sep='\t')
        self.assertFalse(validate_reaction_file(empty_file))
        os.remove(empty_file)

        # Test with invalid file
        invalid_file = "invalid.csv"
        pd.DataFrame({'smarts': ['', 'test'], 'name': ['test', '']}).to_csv(invalid_file, index=False, sep='\t')
        self.assertFalse(validate_reaction_file(invalid_file))
        os.remove(invalid_file)

    def test_read_reaction_file_error(self):
        with self.assertRaises(ValueError):
            read_reaction_file("nonexistent.csv")

if __name__ == '__main__':
    unittest.main() 