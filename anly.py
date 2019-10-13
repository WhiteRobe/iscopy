def build_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input')
    parser.add_argument('--filename')
    parser.add_argument('--summary', default=False)
    parser.add_argument('--same_line_rate', default=None)  # 相同行的判定比例
    parser.add_argument('--template', default=None)
    parser.add_argument('--output', default='./output')
    return parser.parse_args()


def prepare(_args):
    if _args.input is None or not os.path.exists(_args.input):
        print('Please set --input. Example: --input ./demo')
        exit(0)

    if _args.filename is None:
        print('Please set --filename. Example: --filename demo-data.py')
        exit(1)

    if not os.path.exists(_args.output):
        os.mkdir(_args.output)


def batch_match(_args):
    inst = an.BatchAnalyser(_args)
    inst.analyse()
    if str(_args.summary) == 'True':
        inst.summary()
    inst.output()


if __name__ == '__main__':
    import os
    import src.analyser.base as an
    args = build_args()

    prepare(args)
    batch_match(args)
