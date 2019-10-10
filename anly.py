def build_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input')
    parser.add_argument('--filename')
    parser.add_argument('--summary', default=True)
    parser.add_argument('--output', default='./output')
    return parser.parse_args()


if __name__ == '__main__':
    import os
    import src.analyser.base as an
    args = build_args()

    if args.input is None or not os.path.exists(args.input):
        print('Please set --input. Example: --input ./demo')
        exit(0)

    if args.filename is None:
        print('Please set --filename. Example: --filename demo-data.py')
        exit(1)

    if not os.path.exists(args.output):
        os.mkdir(args.output)

    inst = an.BatchAnalyser(args)
    inst.analyse()
    if str(args.summary) == 'True':
        inst.summary()
    inst.output()

