import sys
sys.path.insert(0, '../')

import unittest
import oracles.oracle_qq as oracle_qq

class TestRunScenario1(unittest.TestCase):
    def test_oracle_qq(self):
        qq = oracle_qq.oracle_qq()

        qq.keep('stubs/movielensTest.txt')
        qq.train()
        self.assertEqual(qq.predict('0', '123'), -1)
        self.assertEqual(qq.predict('0', '2'), 4)

if __name__ == '__main__':
    unittest.main()
