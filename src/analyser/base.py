class BaseAnalyser:
    def __init__(self, args):
        self.args = args
        self.template = open(self.args.template, encoding='utf-8').read() if self.args.template is not None else None


class BatchAnalyser(BaseAnalyser):
    def __init__(self, args):
        super(BatchAnalyser, self).__init__(args)
        self.users = {}
        self.result = {}

    def analyse(self):

        self.extract()  # 提取特征

        user_names, user_props = list(self.users.keys()), list(self.users.values())

        self.init_user(user_names)

        for i, user_name in enumerate(user_names):
            print('分析 -- %s -- 的特征重复度:' % user_name)
            for j in range(i + 1, len(user_props)):
                print('\t正在分析: -- %s -- VS. -- %s --' % (user_names[i], user_names[j]))
                da = DualAnalyser(self.args, (user_names[i], user_props[i]), (user_names[j], user_props[j]))
                match_radio, same_line = da.analyse()
                print('\t\t常规匹配指数: %f\n, \t\t宽松相同行数: %d' % (match_radio, same_line))

                self.find_max_match(match_radio, user_name, user_names[j], 'max_match_radio', 'max_match_opponent')
                self.find_max_match(same_line, user_name, user_names[j], 'max_same_line', 'max_same_line_opponent')

            # 相同行比例
            self.result[user_name]['same_line_radio'] = \
                self.result[user_name]['max_same_line'] / self.users[user_name]['total_line'] \
                    if self.users[user_name]['total_line'] > 0 else 0

    def summary(self):
        print('\n分析完毕，输出总体结果：')
        for i in self.users.keys():
            print('  ', i, '\t', self.result[i])

    def output(self):
        import pandas as pd
        result = pd.DataFrame(self.result)
        result.T.to_csv(self.args.output+'/summary.csv', encoding='utf_8_sig')
        print('\n------完成！结果输出到 %s------' % self.args.output+'/summary.csv')

    def init_user(self, user_names):
        for _ in user_names:
            self.result[_] = {}
            self.result[_]['max_match_radio'] = 0
            self.result[_]['max_same_line'] = 0
            self.result[_]['total_line'] = self.users[_]['total_line']

    def extract(self):
        import re
        import os
        from ..extractor.base import BaseExtractor
        for _dir in os.listdir(self.args.input):
            print('提取 -- %s -- 的特征...' % _dir)
            filepath = re.sub(r'/+', '/', self.args.input + '/' + _dir + '/' + self.args.filename)
            if os.path.exists(filepath):
                inst = BaseExtractor(filepath, template=self.template)
                self.users[_dir] = inst.properties()
            else:
                print('\t 没有检测到文件')
                self.result[_dir] = {'err': 'No file detect!'}

    def find_max_match(self, radio, user_name, opponent_name, value_key, opponent_key):
        if radio > self.result[user_name][value_key]:
            self.result[user_name][opponent_key] = opponent_name
            self.result[user_name][value_key] = radio
            self.result[opponent_name][opponent_key] = user_name
            self.result[opponent_name][value_key] = radio


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
        same_line = self.count_same_line()
        return SequenceMatcher(None, self.file1props['code'], self.file2props['code']).quick_ratio(), same_line

    def count_same_line(self):
        from difflib import SequenceMatcher
        same_line = 0
        me_codes = self.file1props['code'].split('\n')
        op_codes = self.file2props['code'].split('\n')
        if self.args.same_line_rate is None:
            for c in me_codes:
                if c in op_codes:
                    same_line += 1
        else:
            assert 0 < self.args.same_line_rate <1
            for me_c in me_codes:
                for op_c in op_codes:
                    if SequenceMatcher(None, me_c, op_c).quick_ratio() > self.args.same_line_rate:
                        same_line += 1
        return same_line
