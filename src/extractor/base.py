class BaseExtractor:
    def __init__(self, filepath, encoding='utf-8', template=None):
        """
        基础分析器
        :param filepath: 要分析的文件路径
        :param encoding: 编码方式
        :param template: 模板文件
        """
        import os
        self.code = open(filepath, "r", encoding=encoding).read()
        self.filename = os.path.split(filepath)[1]
        self.template = template
        self.props = {
            'code': '',
            'total_line': 0,  # 总行数
            'operators': [],  # 操作符
            'variables': [],  # 变量数
            'functions': [],  # 函数列表
            # .... 其它非必须的量
        }

    def properties(self):
        """
        基础分析器
            props = {
                'code': '',
                'total_line': 0,  # 总行数
                'operators': [],  # 操作符
                'variables': [],  # 变量数
                'functions': [],  # 函数列表
                # .... 其它非必须的量
            }
        :return: 返回一些比较基础的提取结果
        """
        import re
        import pygments.lexers
        import pygments.token as tk
        from ..purifier.base import BasePurifier
        self.props = {}

        code = BasePurifier().prune(self.code, self.template)
        lexer = pygments.lexers.guess_lexer_for_filename(self.filename, code)
        tokens = list(lexer.get_tokens(code))

        self.props['code'] = code
        self.props['total_line'] = code.count('\n')
        self.props['operators'] = list(filter(lambda t: t[0] == tk.Operator, tokens))
        self.props['variables'] = list(filter(lambda t: t[0] == tk.Name, tokens))
        self.props['functions'] = list(filter(lambda t: t[0] == tk.Name.Function, tokens))

        # 循环/条件控制数量
        self.props['loop_count(for)'] = len(re.findall('while', code))
        self.props['loop_count(while)'] = len(re.findall('while', code))
        self.props['condition_count(if)'] = len(re.findall('if', code))

        return self.props
