import argparse

class Arguments():
    def __init__(self):
        self._parser = argparse.ArgumentParser(description="Arguments for Fusion")

        self._parser.add_argument("--exp_name", type=str, default="Branch")

        # Dataset arguments
        self._parser.add_argument("--dataset_path" , type=str, default="./data/shapenetcore_partanno_segmentation_benchmark_v0")
        self._parser.add_argument("--batch_size", type=int, default=36)
        self._parser.add_argument("--test_batch_size", type=int, default=1)
        self._parser.add_argument("--point_num", type=int, default=2048)

        self._parser.add_argument("--word_len", type=int, default=10)
        self._parser.add_argument("--vocab_size", type=int, default=15)

        # Training arguments
        self._parser.add_argument("--seed", type=int, default=0)
        self._parser.add_argument("--gpu_id", type=int, default=1)
        self._parser.add_argument("--epochs", type=int, default=1000)
        self._parser.add_argument("--lr", type=float, default=1e-4)
        self._parser.add_argument("--ckpt_path", type=str, default="./model/checkpoints/")
        self._parser.add_argument("--ckpt_save", type=str, default="branch_ckpt_")
        self._parser.add_argument("--ckpt_load", type=str, default="branch_ckpt_1000_Branch.pt")
        # self._parser.add_argument("--ckpt_load", type=str, default=None)
        self._parser.add_argument("--vis_port", type=int, default=8088)
        self._parser.add_argument("--vis_iter", type=int, default=10)
        self._parser.add_argument("--save_iter", type=int, default=10)
        self._parser.add_argument("--D_iter", type=int, default=2)
        self._parser.add_argument("--W_cls", type=float, default=0.4)


        self._parser.add_argument('--visdom_color', type=int, default=4, help='Number of colors for visdom pointcloud visualization. (default:4)')

        # Network arguments
        self._parser.add_argument('--lambdaGP', type=int, default=10, help='Lambda for GP term.')
        self._parser.add_argument('--support', type=int, default=10, help='Support value for TreeGCN loop term.')
        self._parser.add_argument('--DEGREE', type=int, default=[2,  2,  2,   2,   2,   64], nargs='+', help='Upsample degrees for generator.')
        self._parser.add_argument('--G_FEAT', type=int, default=[96, 64, 64,  64,  64,  64, 3], nargs='+', help='Features for generator.')
        self._parser.add_argument('--D_FEAT', type=int, default=[3,  64, 128, 256, 512, 1024], nargs='+', help='Features for discriminator.')


    def parser(self):
        return self._parser
