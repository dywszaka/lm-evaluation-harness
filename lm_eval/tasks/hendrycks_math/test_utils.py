import unittest

import utils


CASES = [
    {
        "actual": "\\begin{pmatrix} \\frac{48}{65} \\\\ \\frac{6}{65} \\end{pmatrix}",
        "expected": "\\begin{pmatrix} 48/65 \\\\ 6/65 \\end{pmatrix}",
    },
    {
        "actual": "\\begin{pmatrix} \\frac{1}{5} & \\frac{4}{5} \\\\ 0 & 1 \\end{pmatrix}",
        "expected": "\\begin{pmatrix} 1/5 & 4/5 \\\\ 0 & 1 \\end{pmatrix}",
    },
    {
        "actual": "-(-8)",
        "expected": "8",
    },
]


class HendrycksMathUtilsTest(unittest.TestCase):
    def test_math_equal_true_when_not_equiv(self):
        for case in CASES:
            with self.subTest(case=case):
                actual = case["actual"]
                expected = case["expected"]

                # Verify that is_equiv returns False
                self.assertFalse(utils.is_equiv(actual, expected), 
                                 msg=f"Expected is_equiv to be False for actual: {actual}, expected: {expected}")
                # Verify that math_equal returns True
                self.assertTrue(utils.math_equal(actual, expected), 
                                msg=f"Expected math_equal to be True for actual: {actual}, expected: {expected}")


if __name__ == "__main__":
    unittest.main()
