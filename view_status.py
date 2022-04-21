import os
import pickle
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Training Log Viewer")
    parser.add_argument("exp_path", type=str, help="path to the experiments directory")
    parser.add_argument("--cur_iter", action='store_true')
    parser.add_argument("--start_iter", type=int, default=0)
    parser.add_argument("--end_iter", type=int, default=-1)
    parser.add_argument("--skip", type=int, default=1)

    args = parser.parse_args()
    base_path = "/workspace/conpro_experiments/experiments"
    log_path = os.path.join(base_path, args.exp_path, "logs/stats.p")

    with open(log_path, 'rb') as pickle_file:
        content = pickle.load(pickle_file)

    if args.cur_iter:
        print('='*40)
        print(f"Current Iteration: {content['losses']['discriminator'][-1][0]}")
        print(f"Current D-Loss: {content['losses']['discriminator'][-1][1]:.4f}")
        print(f"Current G-Loss: {content['losses']['generator'][-1][1]:.4f}")
        print(f"Current Reg-strength: {content['losses']['regularizer'][-1][1]:.4f}")
        print('='*40)

    else:
        its, ds, gs, regs = [], [], [], []
        left, right, step = args.start_iter, args.end_iter, args.skip
        for it in range(left, right+1, step):
            its += [it]
            ds += [f"{content['losses']['discriminator'][it][1]:.4f}"]
            gs += [f"{content['losses']['generator'][it][1]:.4f}"]
            # regs += [f"{content['losses']['regularizer'][it][1]:.4f}"]

        print(f"its: {its}")
        print(f"dis: {ds}")
        print(f"gen: {gs}")
        # print(f"reg: {regs}")
