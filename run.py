import argparse
import dataset
import safe_action
import train



def run(args):
    # train_safe_action.train(train_dl, test_dl, args)
    # dataloader = dataset.create_non_appended_dataloader(args)
    train.run(args)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--assets_dir', type=str, default='C:\\Users\\armin\\P920_output')

    parser.add_argument('--env', type=str, default='None')
    parser.add_argument('--minari_env', type=str, default='D4RL/door/expert-v2')
    parser.add_argument('--state_dim', type=int, default=39)
    parser.add_argument('--action_dim', type=int, default=28)
    
    parser.add_argument('--synthetic_size', type=int, default=1000000)
    parser.add_argument('--safe_action_num_epochs', type=int, default=200)
    parser.add_argument('--safe_action_hidden_dim', type=int, default=256)
    parser.add_argument('--safe_action_lr', type=float, default=0.0001)
    parser.add_argument('--safe_action_bs', type=int, default=512)

    parser.add_argument('--base_algo', type=str, default='SAC')
    parser.add_argument('--base_algo_bs', type=int, default=1024)

    args = parser.parse_args()

    # dataset.create(args)
    # safe_action.train(args)
    run(args)


if __name__ == '__main__':
    main()