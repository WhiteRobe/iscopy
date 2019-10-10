class BasePurifier:
    def prune(self, code):
        """
        过滤代码中的各类可省略内容
        :param code:
        :return:
        """
        import re
        # 去除行注释
        code = re.sub(r'^\s+|#.*|//.*', '', code, flags=re.M)
        # 去除块注释
        code = re.sub(r'/\*.*\*/|""".*"""', '', code, flags=re.DOTALL)
        # 去除空格
        code = re.sub(r' *', '', code)
        # 去除空白行
        code = re.sub(r'^\n+|\r+$', '', code)
        return code

    def turn(self,  _str):
        _str = _str.replace('{', '\\{')
        _str = _str.replace('}', '\\}')
        _str = _str.replace('(', '\\(')
        _str = _str.replace(')', '\\)')
        _str = _str.replace(';', '\\;')
        _str = _str.replace('[', '\\[')
        _str = _str.replace(']', '\\]')
        _str = _str.replace('+', '\\+')
        _str = _str.replace('*', '\\*')
        _str = _str.replace('?', '\\?')
        return _str
