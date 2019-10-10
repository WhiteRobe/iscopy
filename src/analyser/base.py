class BaseAnalyser:
    def __init__(self, args):
        self.args = args


class BatchAnalyser(BaseAnalyser):
    def __init__(self, args):
        super(BatchAnalyser, self).__init__(args)
        self.users = {}
        self.result = {}

    def analyse(self):
        import re
        import os
        from ..extractor.base import BaseExtractor
        for _dir in os.listdir(self.args.input):
            filepath = re.sub(r'/+', '/', self.args.input + '/' + _dir + '/' + self.args.filename)
            inst = BaseExtractor(filepath)
            self.users[_dir] = inst.properties()

        user_names = list(self.users.keys())
        user_props = list(self.users.values())

        for _ in user_names:
            self.result[_] = {}
            self.result[_]['max_match_radio'] = 0

        for i, user_name in enumerate(user_names):
            print('提取 -- %s -- 的特征' % user_name)
            for j in range(i + 1, len(user_props)):
                print('\t正在分析: -- %s -- VS. -- %s --' % (user_names[i], user_names[j]))
                da = DualAnalyser(self.args, (user_names[i], user_props[i]), (user_names[j], user_props[j]))
                match_radio = da.analyse()
                print('\t\t常规匹配指数: %f' % match_radio)

                if match_radio > self.result[user_name]['max_match_radio']:
                    self.result[user_name]['max_match_opponent'] = user_names[j]
                    self.result[user_name]['max_match_radio'] = match_radio
                    self.result[user_names[j]]['max_match_opponent'] = user_name
                    self.result[user_names[j]]['max_match_radio'] = match_radio

    def summary(self):
        print('\n分析完毕，输出总体结果：')
        for i in self.users.keys():
            print('  ', i, '\t', self.result[i])

    def output(self):
        import pandas as pd
        result = pd.DataFrame(self.result)
        result.T.to_csv(self.args.output+'/summary.csv', encoding='utf_8_sig')


class DualAnalyser(BaseAnalyser):
    def __init__(self, args, file1, file2):
        super(DualAnalyser, self).__init__(args)
        assert isinstance(file1, tuple) and isinstance(file2, tuple)
        assert isinstance(file1[0], str) and isinstance(file2[0], str)
        self.file1name = file1[0]
        self.file2name = file2[0]
        self.file1props = file1[1]
        self.file2props = file2[1]

    def analyse(self):
        from difflib import SequenceMatcher
        return SequenceMatcher(None, self.file1props['code'], self.file2props['code']).quick_ratio()
