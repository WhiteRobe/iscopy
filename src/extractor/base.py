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

    def properties(self):
        """
        基础分析器
        :return: 返回一些比较基础的提取结果
        """
        import re
        import pygments.lexers
        import pygments.token as tk
        from ..purifier.base import BasePurifier
        props = {
            'code': '',
            'total_line': 0,  # 总行数
            'operators': [],  # 操作符
            'variables': [],  # 变量数
            'functions': [],  # 函数列表
            # .... 其它非必须的量
        }

        code = BasePurifier().prune(self.code, self.template)
        lexer = pygments.lexers.guess_lexer_for_filename(self.filename, code)
        tokens = list(lexer.get_tokens(code))

        props['code'] = code
        props['total_line'] = code.count('\n')
        props['operators'] = list(filter(lambda t: t[0] == tk.Operator, tokens))
        props['variables'] = list(filter(lambda t: t[0] == tk.Name, tokens))
        props['functions'] = list(filter(lambda t: t[0] == tk.Name.Function, tokens))

        # 循环/条件控制数量
        props['loop_count(for)'] = len(re.findall('while', code))
        props['loop_count(while)'] = len(re.findall('while', code))
        props['condition_count(if)'] = len(re.findall('if', code))

        return props
