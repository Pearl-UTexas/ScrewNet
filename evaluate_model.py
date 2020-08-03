import argparse
import copy
import os
import sys

import matplotlib
import numpy as np
import torch
from ScrewNet.dataset import ArticulationDataset, RigidTransformDataset
from ScrewNet.models import ScrewNet, ScrewNet_2imgs, ScrewNet_NoLSTM
from ScrewNet.utils import distance_bw_plucker_lines, difference_between_quaternions_tensors, interpret_labels_ours
from GeneralizingKinematics.magic.mixture import mdn
from GeneralizingKinematics.magic.mixture.dataset import MixtureDataset
from GeneralizingKinematics.magic.mixture.models import KinematicMDNv3
from GeneralizingKinematics.magic.mixture.utils import *
from matplotlib.ticker import FuncFormatter

matplotlib.use('Agg')
from matplotlib import pyplot as plt


def to_percent(y, position):
    # Ignore the passed in position. This has the effect of scaling the default
    # tick locations.
    global percent_scale
    s = str(round(100 * y / percent_scale, 3))

    # The percent symbol needs escaping in latex
    if matplotlib.rcParams['text.usetex'] is True:
        return s + r'$\%$'
    else:
        return s + '%'


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test model for articulated object dataset.")
    parser.add_argument('--model-dir', type=str, default='models/')
    parser.add_argument('--model-name', type=str, default='test_lstm')
    parser.add_argument('--test-dir', type=str, default='../data/test/microwave/')
    parser.add_argument('--output-dir', type=str, default='./plots/')
    parser.add_argument('--ntest', type=int, default=1, help='number of test samples (n_object_instants)')
    parser.add_argument('--ndof', type=int, default=1, help='how many degrees of freedom in the object class?')
    parser.add_argument('--batch', type=int, default=128, help='batch size')
    parser.add_argument('--nwork', type=int, default=8, help='num_workers')
    parser.add_argument('--device', type=int, default=0, help='cuda device')
    parser.add_argument('--dual-quat', action='store_true', default=False, help='Dual quaternion representation or not')
    parser.add_argument('--model-type', type=str, default='lstm', help='screw, noLSTM, 2imgs, l2, baseline')
    parser.add_argument('--load-wts', action='store_true', default=False, help='Should load model wts from prior run?')
    parser.add_argument('--obj', type=str, default='microwave')
    args = parser.parse_args()

    # Dataset for common axis
    all_ori_err_mean = torch.empty(0)
    all_ori_err_std = torch.empty(0)
    all_dist_err_mean = torch.empty(0)
    all_dist_err_std = torch.empty(0)

    output_dir = os.path.join(os.path.abspath(args.output_dir), args.model_type, args.model_name)
    os.makedirs(output_dir, exist_ok=True)

    if torch.cuda.is_available():
        device = torch.device(args.device)
    else:
        device = torch.device('cpu')

    # Plotting Histograms as percentages
    formatter = FuncFormatter(to_percent)
    global percent_scale

    if args.model_type == 'baseline':
        print("Testing Model: Abbattematteo et al.")
        bounds = np.load(os.path.join(args.test_dir, 'bounds.npy'))
        keep_columns = np.load(os.path.abspath('../GeneralizingKinematics/keep_columns_' + args.obj + '.npy'))
        one_columns = np.load(os.path.abspath('../GeneralizingKinematics/one_columns_' + args.obj + '.npy'))
        testset = MixtureDataset(args.ntest,
                                 args.test_dir,
                                 n_dof=args.ndof,
                                 normalize=True,
                                 bounds=bounds,
                                 keep_columns=keep_columns,
                                 one_columns=one_columns
                                 )

        best_model = KinematicMDNv3(n_gaussians=20,
                                    out_features=testset.labels.shape[1])
        best_model.load_state_dict(torch.load(os.path.join(args.model_dir, args.model_name + '.net')))
        best_model.float().to(device)
        best_model.eval()

        testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch,
                                                 shuffle=False, num_workers=args.nwork,
                                                 pin_memory=True)

        all_labels = torch.zeros(args.ntest, testloader.dataset.full_labels.shape[1])
        all_output = torch.zeros(args.ntest, testloader.dataset.full_labels.shape[1])

        with torch.no_grad():
            for i, X in enumerate(testloader):
                depth, labels = X['depth'].to(device), X['label'].to(device)
                pi, sigma, mu = best_model(depth)
                output = mdn.sample(pi, sigma, mu)

                # expand labels and output back to full dataset size
                output = expand_labels(testloader.dataset.full_labels,
                                       output,
                                       testloader.dataset.keep_columns,
                                       testloader.dataset.one_columns)

                labels = expand_labels(testloader.dataset.full_labels,
                                       labels,
                                       testloader.dataset.keep_columns,
                                       testloader.dataset.one_columns)

                all_labels[i * labels.shape[0]:(i + 1) * labels.shape[0], :] = labels
                all_output[i * labels.shape[0]:(i + 1) * labels.shape[0], :] = output

        # interpret and convert to real.
        ground_truth_params = convert_dict_to_real(interpret_labels(all_labels, args.ndof), bounds, args.ndof)
        param_dict = convert_dict_to_real(interpret_labels(all_output, args.ndof), bounds, args.ndof)

        n_imgs_per_obj = 16
        axis_len = 7
        config_len = 1

        real_axis = ground_truth_params['axis'].view(-1, args.ndof, n_imgs_per_obj, axis_len)  # Axis len 7
        real_net_axis = param_dict['axis'].view(-1, args.ndof, n_imgs_per_obj, axis_len)

        all_dist_err_std, all_dist_err_mean = torch.std_mean(
            torch.norm(real_axis[:, :, :, :3] - real_net_axis[:, :, :, :3], dim=-1), dim=-1)
        all_dist_err_std = all_dist_err_std.squeeze_().cpu()
        all_dist_err_mean = all_dist_err_mean.squeeze_().cpu()

        all_ori_err_std, all_ori_err_mean = torch.std_mean(
            difference_between_quaternions_tensors(real_axis[:, :, :, 3:7], real_net_axis[:, :, :, 3:7]), dim=-1)
        all_ori_err_mean = all_ori_err_mean.squeeze_().cpu()
        all_ori_err_std = all_ori_err_std.squeeze_().cpu()

        # Configuration error
        real_configs = ground_truth_params['config'].view(-1, args.ndof, n_imgs_per_obj, config_len)  # Config len 1
        net_configs = param_dict['config'].view(-1, args.ndof, n_imgs_per_obj, config_len)

        all_q_err_std, all_q_err_mean = torch.std_mean(
            torch.norm(real_configs[:, :, :, :] - net_configs[:, :, :, :], dim=-1), dim=-1)
        all_q_err_std = all_q_err_std.squeeze_().cpu()
        all_q_err_mean = all_q_err_mean.squeeze_().cpu()

        x_axis = np.arange(np.shape(all_q_err_mean)[0])
        fig = plt.figure(3)
        plt.errorbar(x_axis, all_q_err_mean.numpy(), all_q_err_std.numpy(), capsize=3., capthick=1., ls='none')
        plt.xlabel("Test object number")
        plt.ylabel("Error in Config")
        plt.title("Test error in Configurations")
        plt.tight_layout()
        plt.savefig(output_dir + '/config_err.png')
        plt.close(fig)

        fig = plt.figure(31)
        data = copy.copy(all_q_err_mean.numpy())
        if args.obj in ['drawer']:
            data *= 100.
            binwidth = 0.5
            title = "Histogram of mean test errors in d"
            x_label = "Error (cm)"
        else:
            binwidth = 0.05
            x_label = "Error (rad)"
            title = "Histogram of mean test errors in theta"

        plt.hist(data, bins=np.arange(0., data.max() + binwidth, binwidth), density=True)
        percent_scale = 1 / binwidth
        plt.gca().yaxis.set_major_formatter(formatter)
        plt.xlabel(x_label)
        plt.ylabel("Percentage of test objects")
        plt.title(title)
        plt.tight_layout()
        plt.savefig(output_dir + '/config_err_hist.png')
        plt.close(fig)

        s_data = {'labels': all_labels.cpu().numpy(), 'predictions': all_output.cpu().numpy(),
                  'ori_err_mean': all_ori_err_mean.numpy(), 'ori_err_std': all_ori_err_std.numpy(),
                  'dist_err_mean': all_dist_err_mean.numpy(), 'dist_err_std': all_dist_err_std.numpy(),
                  'q_err_mean': all_q_err_mean.numpy(), 'q_err_std': all_q_err_std.numpy()}

    else:
        if args.model_type == '2imgs':
            print("Testing Model: ScrewNet_2imgs")
            best_model = ScrewNet_2imgs(n_output=8)
            testset = RigidTransformDataset(args.ntest, args.test_dir)

        elif args.model_type == 'noLSTM':
            print("Testing Model: ScrewNet_noLSTM")
            best_model = ScrewNet_NoLSTM(seq_len=16, fc_replace_lstm_dim=1000, n_output=8)
            testset = ArticulationDataset(args.ntest, args.test_dir)

        else:
            print("Testing ScrewNet")
            best_model = ScrewNet(lstm_hidden_dim=1000, n_lstm_hidden_layers=1, n_output=8)
            testset = ArticulationDataset(args.ntest, args.test_dir)

        best_model.load_state_dict(torch.load(os.path.join(args.model_dir, args.model_name + '.net')))
        best_model.float().to(device)
        best_model.eval()

        testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch, shuffle=False, num_workers=args.nwork,
                                                 pin_memory=True)

        all_q_err_mean = torch.empty(0)
        all_q_err_std = torch.empty(0)
        all_d_err_mean = torch.empty(0)
        all_d_err_std = torch.empty(0)

        obj_idxs = torch.empty(0)  # Recording object indexes for analysis

        # Data collection for post-processing
        all_labels = torch.empty(0)
        all_preds = torch.empty(0)

        with torch.no_grad():
            for X in testloader:
                depth, labels = X['depth'].to(device), X['label'].to(device)
                y_pred = best_model(depth)
                y_pred = y_pred.view(y_pred.size(0), -1, 8)
                y_pred = y_pred[:, 1:, :]

                labels = interpret_labels_ours(labels, testset.normalization_factor)  # Scaling m appropriately
                y_pred = interpret_labels_ours(y_pred, testset.normalization_factor)

                # Orientation error
                ori_err_std, ori_err_mean = torch.std_mean(torch.acos(
                    torch.mul(labels[:, :, :3], y_pred[:, :, :3]).sum(dim=-1) / (
                            torch.norm(labels[:, :, :3], dim=-1) * torch.norm(y_pred[:, :, :3], dim=-1))), dim=-1)
                all_ori_err_mean = torch.cat((all_ori_err_mean, ori_err_mean.cpu()))
                all_ori_err_std = torch.cat((all_ori_err_std, ori_err_std.cpu()))

                # Distance b/w plucker lines in cm
                dist_err_std, dist_err_mean = torch.std_mean(distance_bw_plucker_lines(labels, y_pred), dim=-1)
                all_dist_err_mean = torch.cat((all_dist_err_mean, dist_err_mean.cpu()))
                all_dist_err_std = torch.cat((all_dist_err_std, dist_err_std.cpu()))

                # Configurational errors
                q_err_std, q_err_mean = torch.std_mean(torch.abs(labels[:, :, 6] - y_pred[:, :, 6]), dim=-1)
                all_q_err_mean = torch.cat((all_q_err_mean, q_err_mean.cpu()))
                all_q_err_std = torch.cat((all_q_err_std, q_err_std.cpu()))

                d_err_std, d_err_mean = torch.std_mean(torch.abs(labels[:, :, 7] - y_pred[:, :, 7]), dim=-1)
                all_d_err_mean = torch.cat((all_d_err_std, d_err_mean.cpu()))
                all_d_err_std = torch.cat((all_d_err_std, d_err_std.cpu()))

                # Data for post-processing
                all_labels = torch.cat((all_labels, labels.cpu()))
                all_preds = torch.cat((all_preds, y_pred.cpu()))

        # Sort objects as per the idxs
        x_axis = np.arange(all_q_err_mean.size(0))

        fig = plt.figure(3)
        plt.errorbar(x_axis, all_q_err_mean.numpy(), all_q_err_std.numpy(), capsize=3., capthick=1., ls='none')
        plt.xlabel("Test object number")
        plt.ylabel("Error (rad)")
        plt.title("Test error in theta")
        plt.tight_layout()
        plt.savefig(output_dir + '/theta_err.png')
        plt.close(fig)

        fig = plt.figure(31)
        data = all_q_err_mean.numpy()
        binwidth = 0.005
        plt.hist(data, bins=np.arange(0., max(data) + binwidth, binwidth), density=True)
        percent_scale = 1 / binwidth
        plt.gca().yaxis.set_major_formatter(formatter)
        plt.xlabel("Error (rad)")
        plt.ylabel("Percentage of test objects")
        plt.title("Histogram of mean test errors in theta")
        plt.tight_layout()
        plt.savefig(output_dir + '/theta_err_hist.png')
        plt.close(fig)

        fig = plt.figure(4)
        plt.errorbar(x_axis, all_d_err_mean.numpy() * 100., all_d_err_std.numpy() * 100., capsize=3., capthick=1.,
                     ls='none')
        plt.xlabel("Test object number")
        plt.ylabel("Error (cm)")
        plt.title("Test error in d")
        plt.tight_layout()
        plt.savefig(output_dir + '/d_err.png')
        plt.close(fig)

        fig = plt.figure(41)
        data = copy.copy(all_d_err_mean.numpy()) * 100.
        binwidth = 0.5
        plt.hist(data, bins=np.arange(0., max(data) + binwidth, binwidth), density=True)
        percent_scale = 1 / binwidth
        plt.gca().yaxis.set_major_formatter(formatter)
        plt.xlabel("Error (cm)")
        plt.ylabel("Percentage of test objects")
        plt.title("Histogram of mean test errors in d")
        plt.tight_layout()
        plt.savefig(output_dir + '/d_err_hist.png')
        plt.close(fig)

        s_data = {'labels': all_labels.numpy(), 'predictions': all_preds.numpy(),
                  'ori_err_mean': all_ori_err_mean.numpy(), 'ori_err_std': all_ori_err_std.numpy(),
                  'dist_err_mean': all_dist_err_mean.numpy(), 'dist_err_std': all_dist_err_std.numpy(),
                  'theta_err_mean': all_q_err_mean.numpy(), 'theta_err_std': all_q_err_std.numpy(),
                  'd_err_mean': all_d_err_mean.numpy(), 'd_err_std': all_d_err_std.numpy()}

    """ Common Plots"""
    # Plot variation of screw axis
    x_axis = np.arange(all_ori_err_mean.size(0))

    fig = plt.figure(1)
    plt.errorbar(x_axis, all_ori_err_mean.numpy(), all_ori_err_std.numpy(), marker='o', mfc='blue', ms=4., capsize=3.,
                 capthick=1., ls='none')
    plt.xlabel("Test object number")
    plt.ylabel("Orientation error (rad)")
    plt.title("Test error in screw axis orientation")
    plt.tight_layout()
    plt.savefig(output_dir + '/orientation_test_error.png')
    plt.close(fig)

    fig = plt.figure(11)
    data = all_ori_err_mean.numpy()
    binwidth = 0.05
    plt.hist(data, bins=np.arange(0., data.max() + binwidth, binwidth), weights=np.ones(len(data)) / len(data),
             density=True)
    percent_scale = 1 / binwidth
    plt.gca().yaxis.set_major_formatter(formatter)
    plt.xlabel("Orientation error (rad)")
    plt.ylabel("Percentage of test objects")
    plt.title("Histogram of mean test errors in screw axis orientation")
    plt.tight_layout()
    plt.savefig(output_dir + '/orientation_test_error_hist.png')
    plt.close(fig)

    fig = plt.figure(2)
    plt.errorbar(x_axis, all_dist_err_mean.numpy() * 100., all_dist_err_std.numpy() * 100.,
                 marker='o', ms=4, mfc='blue', capsize=3., capthick=1., ls='none')
    plt.xlabel("Test object number")
    plt.ylabel("Spatial distance error (cm)")
    plt.title("Test error in spatial distance")
    plt.tight_layout()
    plt.savefig(output_dir + '/distance_test_error.png')
    plt.close(fig)

    fig = plt.figure(21)
    data = copy.copy(all_dist_err_mean.numpy()) * 100.
    binwidth = 1.
    plt.hist(data, bins=np.arange(0., data.max() + binwidth, binwidth), density=True)
    percent_scale = 1 / binwidth
    plt.gca().yaxis.set_major_formatter(formatter)
    plt.xlabel("Spatial distance error (cm)")
    plt.ylabel("Percentage of test objects")
    plt.title("Histogram of mean test errors in spatial distance")
    plt.tight_layout()
    plt.savefig(output_dir + '/distance_test_error_hist.png')
    plt.close(fig)

    print("Saved plots in directory {}".format(output_dir))

    # Storing data for particle filter analysis
    import pickle

    with open(output_dir + '/test_prediction_data.pkl', 'wb') as handle:
        pickle.dump(s_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
