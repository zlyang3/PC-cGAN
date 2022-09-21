import torch
import torch.optim as optim
import torch.nn as nn

from data.dataset import ShapeNetDataset
from model.model import Generator, Discriminator_with_Classifier, Discriminator
from model.gradient_penalty import GradientPenalty
from evaluation.FPD import calculate_fpd

from arguments import Arguments

import time
import visdom
import numpy as np

class cGAN():
    def __init__(self, args):
        self.args = args
        # ------------------------------------------------Dataset---------------------------------------------- #
        self.data = ShapeNetDataset(root=args.dataset_path, num_points=args.point_num)
        # self.data = ShapeNetDataset(num_points=args.point_num)
        self.dataLoader = torch.utils.data.DataLoader(self.data, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=4, drop_last=True)
        print("Training Dataset : {} prepared.".format(len(self.data)))
        # ----------------------------------------------------------------------------------------------------- #

        # -------------------------------------------------Module---------------------------------------------- #
        self.G = Generator(batch_size=args.batch_size, features=args.G_FEAT, degrees=args.DEGREE, support=args.support).to(args.device)
        self.D = Discriminator_with_Classifier(features=args.D_FEAT).to(args.device)
        self.D_pre = Discriminator(features=args.D_FEAT).to(args.device)

        self.optimizerG = optim.Adam(self.G.parameters(), lr=args.lr, betas=(0, 0.99))
        self.optimizerD = optim.Adam(self.D.parameters(), lr=args.lr, betas=(0, 0.99))
        self.optimizerD_pre = optim.Adam(self.D_pre.parameters(), lr=args.lr, betas=(0, 0.99))

        self.criterion = nn.CrossEntropyLoss()

        self.GP = GradientPenalty(args.lambdaGP, gamma=1, device=args.device)
        print("Network prepared.")
        # ----------------------------------------------------------------------------------------------------- #

        # ---------------------------------------------Visualization------------------------------------------- #
        self.vis = visdom.Visdom(port=args.vis_port, env=args.exp_name)
        assert self.vis.check_connection()
        print("Visdom connected.")
        # ----------------------------------------------------------------------------------------------------- #

    def run(self, save_ckpt=None, load_ckpt=None):
        loss_log = {}
        metric = {}
        if load_ckpt is None:
            epoch_log = 0
            iter_log = 0

            loss_log = {'G_loss': [], 'D_loss': [], 'D_pre_loss': []}
            loss_legend = list(loss_log.keys())

            metric = {'FPD': []}
        else:
            checkpoint = torch.load(load_ckpt)
            self.D.load_state_dict(checkpoint['D_state_dict'])
            self.D_pre.load_state_dict(checkpoint['D_pre_state_dict'])
            self.G.load_state_dict(checkpoint['G_state_dict'])

            epoch_log = checkpoint['epoch'] + 1
            iter_log = checkpoint['iter']

            loss_log['G_loss'] = checkpoint['G_loss']
            loss_log['D_loss'] = checkpoint['D_loss']
            loss_log['D_pre_loss'] = checkpoint['D_pre_loss']
            loss_legend = list(loss_log.keys())

            metric['FPD'] = checkpoint['FPD']

            print("Checkpoint loaded.")

        for epoch in range(epoch_log, self.args.epochs):
            for _iter, data in enumerate(self.dataLoader, iter_log):
                # Start Time
                start_time = time.time()
                point, _, cls= data
                point = point.to(args.device).float()
                cls = cls.to(args.device)

                bsz = point.size(0)

                # -------------------- Discriminator -------------------- #
                for d_iter in range(self.args.D_iter):
                    self.D.zero_grad()
                    self.D_pre.zero_grad()

                    z = torch.randn(bsz, 1, 96).to(args.device)
                    tree = [z]

                    with torch.no_grad():
                        fake_point_pre, fake_point = self.G(tree, cls)

                    D_real, D_real_pred_cls = self.D(point)
                    D_realm = D_real.mean()
                    D_fake, _ = self.D(fake_point)
                    D_fakem = D_fake.mean()
                    gp_loss = self.GP(self.D, point.data, fake_point.data)
                    D_cls_loss = self.criterion(D_real_pred_cls, cls.squeeze())
                    d_loss = -D_realm + D_fakem
                    d_loss_gp = d_loss + gp_loss + D_cls_loss

                    # d_loss = -D_realm + D_fakem
                    d_loss_gp.backward()
                    self.optimizerD.step()

                    D_real = self.D_pre(point)
                    D_realm = D_real.mean()
                    D_fake_pre = self.D_pre(fake_point_pre)
                    D_fake_prem = D_fake_pre.mean()
                    gp_loss = self.GP(self.D_pre, point.data, fake_point_pre.data)
                    d_loss_pre = -D_realm + D_fake_prem
                    d_loss_gp = d_loss_pre + gp_loss

                    d_loss_gp.backward()
                    self.optimizerD_pre.step()

                loss_log['D_pre_loss'].append(d_loss_pre.item())
                loss_log['D_loss'].append(d_loss.item())

                # ---------------------- Generator ---------------------- #
                self.G.zero_grad()

                z = torch.randn(bsz, 1, 96).to(args.device)
                tree = [z]

                fake_point_pre, fake_point = self.G(tree, cls)
                G_fake_pre = self.D_pre(fake_point_pre)
                G_fake, G_fake_pred_cls = self.D(fake_point)
                G_cls_loss = self.criterion(G_fake_pred_cls, cls.squeeze())

                G_fakem = G_fake.mean() + G_fake_pre.mean()
                # G_fakem = G_fake.mean()

                g_loss = -G_fakem + 0.5*G_cls_loss
                g_loss.backward()
                self.optimizerG.step()

                loss_log['G_loss'].append(g_loss.item())

                # --------------------- Visualization -------------------- #
                print("[Epoch/Iter] ", "{:3} / {:3}".format(epoch, _iter),
                      "[ D_Loss ] ", "{: 7.6f}".format(d_loss),
                      "[ D_pre_Loss ] ", "{: 7.6f}".format(d_loss_pre),
                      "[ G_Loss ] ", "{: 7.6f}".format(g_loss),
                      "[ Time ] ", "{:4.2f}s".format(time.time()-start_time))

                if _iter % self.args.vis_iter == 0:
                    pregenerated_point = self.G.getPrePointCloud()
                    generated_point = self.G.getPointcloud()
                    generated_cls = self.G.getLabel()
                    cls_name = list(self.data.classes)[generated_cls]
                    plot_X = np.stack([np.arange(len(loss_log[legend])) for legend in loss_legend], 1)
                    plot_Y = np.stack([np.array(loss_log[legend]) for legend in loss_legend], 1)

                    tick_max = np.round(torch.abs(torch.cat([pregenerated_point, generated_point, point[-1]], dim=0)).max(0)[0].cpu().detach().numpy() * 10 +1) /10
                    tick_max = tick_max[[2,0,1]].tolist()


                    self.vis.scatter(X=generated_point[:,torch.LongTensor([2,0,1])], win=1,
                                     opts={'title': cls_name, 'markersize': 2, 'webgl': True, "xtickmin": -tick_max[0], "ytickmin": -tick_max[1],"ztickmin": -tick_max[2],"xtickmax": tick_max[0],"ytickmax": tick_max[1],"ztickmax": tick_max[2]})
                    self.vis.scatter(X=pregenerated_point[:,torch.LongTensor([2,0,1])], win=2,
                                     opts={'title': cls_name+"(pre)", 'markersize': 2, 'webgl': True, "xtickmin": -tick_max[0], "ytickmin": -tick_max[1],"ztickmin": -tick_max[2],"xtickmax": tick_max[0],"ytickmax": tick_max[1],"ztickmax": tick_max[2]})
                    self.vis.scatter(X=point[-1][:,torch.LongTensor([2,0,1])], win=3,
                                     opts={'title': cls_name+"(real)", 'markersize': 2, 'webgl': True, "xtickmin": -tick_max[0], "ytickmin": -tick_max[1],"ztickmin": -tick_max[2],"xtickmax": tick_max[0],"ytickmax": tick_max[1],"ztickmax": tick_max[2]})

                    self.vis.line(X=plot_X, Y=plot_Y, win=4,
                                  opts={'title': 'TreeGAN Loss', 'legend': loss_legend, 'xlabel': 'Iteration', 'ylabel': 'Loss'})

                    if len(metric['FPD']) > 0:
                        self.vis.line(X=np.arange(len(metric['FPD'])), Y=np.array(metric['FPD']), win=5,
                                      opts={'title': "Frechet Pointcloud Distance", 'legend': ["FPD best : {}".format(np.min(metric['FPD']))]})

                    print('Figures are saved.')
            # ---------------------- Save checkpoint --------------------- #
            if epoch % self.args.save_iter == 0 and not save_ckpt == None:
                torch.save({
                        'epoch': epoch,
                        'iter': _iter,
                        'D_state_dict': self.D.state_dict(),
                        'D_pre_state_dict': self.D_pre.state_dict(),
                        'G_state_dict': self.G.state_dict(),
                        'D_loss': loss_log['D_loss'],
                        'D_pre_loss': loss_log['D_pre_loss'],
                        'G_loss': loss_log['G_loss'],
                        'FPD': metric['FPD']
                }, save_ckpt+str(epoch)+'_Branch.pt')

                print('Checkpoint is saved.')

            # ---------------- Frechet Pointcloud Distance --------------- #
            if epoch % 1 == 0:
                fake_pointclouds = torch.Tensor([])
                for i in range(100): # batch_size * 100
                    z = torch.randn(cls.size(0), 1, 96).to(self.args.device)
                    tree = [z]
                    with torch.no_grad():
                        sample = self.G(tree, cls).cpu()
                    fake_pointclouds = torch.cat((fake_pointclouds, sample), dim=0)

                fpd = calculate_fpd(fake_pointclouds, batch_size=100, dims=1808, device=self.args.device)
                metric['FPD'].append(fpd)
                print('[{:4} Epoch] Frechet Pointcloud Distance <<< {:.10f} >>>'.format(epoch, fpd))

                class_name = args.class_choice if args.class_choice is not None else 'all'
                torch.save(fake_pointclouds, './model/generated/branchGCN_{}_{}.pt'.format(str(epoch), class_name))
                del fake_pointclouds



if __name__ == '__main__':
    args = Arguments().parser().parse_args()

    args.device = torch.device('cuda:'+str(args.gpu_id) if torch.cuda.is_available() else 'cpu')

    SAVE_CHECKPOINT = args.ckpt_path + args.ckpt_save if args.ckpt_save is not None else None
    LOAD_CHECKPOINT = args.ckpt_path + args.ckpt_load if args.ckpt_load is not None else None

    model = cGAN(args)
    model.run(save_ckpt=SAVE_CHECKPOINT, load_ckpt=LOAD_CHECKPOINT)

