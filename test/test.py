import unittest


class MyTestCase(unittest.TestCase):

    def build_args(self):
        class _Args:
            def __init__(self):
                self.input = '../demo'
                self.filename = 'demo-data.py'
                self.summary = False
                self.template = '../demo/template/demo-data.py'
                self.output = '../output'

        return _Args()

    def test_anly(self):
        import src.analyser.base as an
        import os
        import pandas as pd
        args = self.build_args()

        if os.path.exists(args.output + '/summary.csv'):  # clean
            os.remove(args.output + '/summary.csv')

        if not os.path.exists(args.output):
            os.mkdir(args.output)

        inst = an.BatchAnalyser(args)
        inst.analyse()
        inst.summary()
        inst.output()

        self.assertTrue(os.path.exists(args.output))
        self.assertEqual(pd.read_csv(args.output + '/summary.csv').shape, (8, 7))


if __name__ == '__main__':
    unittest.main()
