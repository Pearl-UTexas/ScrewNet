import torch
from ScrewNet.utils import distance_bw_plucker_lines, \
    orientation_difference_bw_plucker_lines, theta_config_error, d_config_error


def articulation_lstm_loss_spatial_distance(pred, target, wt_on_ortho=1.):
    """ Based on Spatial distance
        Input shapes: Batch X Objects X images
    """
    pred = pred.view(pred.size(0), -1, 8)[:, 1:, :]  # We don't need the first row as it is for single image
    # pred = expand_labels(pred)  # Adding 3rd dimension to m, if needed

    # Spatial Distance loss
    dist_err = orientation_difference_bw_plucker_lines(target, pred) ** 2 + \
               2. * distance_bw_plucker_lines(target, pred) ** 2

    # Configuration Loss
    conf_err = theta_config_error(target, pred) ** 2 + d_config_error(target, pred) ** 2

    err = dist_err + conf_err
    loss = torch.mean(err)

    # Ensure l_hat has norm 1.
    loss += torch.mean((torch.norm(pred[:, :, :3], dim=-1) - 1.) ** 2)

    # Ensure orthogonality between l_hat and m
    loss += wt_on_ortho * torch.mean(torch.abs(torch.sum(torch.mul(pred[:, :, :3], pred[:, :, 3:6]), dim=-1)))

    if torch.isnan(loss):
        print("target: Min: {},  Max{}".format(target.min(), target.max()))
        print("Prediction: Min: {},  Max{}".format(pred.min(), pred.max()))
        print("L2 error: {}".format(torch.mean((target - pred) ** 2)))
        print("Distance loss:{}".format(torch.mean(orientation_difference_bw_plucker_lines(target, pred) ** 2)))
        print("Orientation loss:{}".format(torch.mean(distance_bw_plucker_lines(target, pred) ** 2)))
        print("Configuration loss:{}".format(torch.mean(conf_err)))

    return loss


def articulation_lstm_loss_spatial_distance_RT(pred, target, wt_on_ortho=1.):
    """ Based on Spatial distance
        Input shapes: Batch X 8
    """
    pred = pred.unsqueeze_(1)
    target = target.unsqueeze_(1)
    # pred = expand_labels(pred)  # Adding 3rd dimension to m, if needed

    # Spatial Distance loss
    dist_err = orientation_difference_bw_plucker_lines(target, pred) ** 2 + \
               2. * distance_bw_plucker_lines(target, pred) ** 2

    # Configuration Loss
    conf_err = theta_config_error(target, pred) ** 2 + d_config_error(target, pred) ** 2

    err = dist_err + conf_err
    loss = torch.mean(err)

    # Ensure l_hat has norm 1.
    loss += torch.mean((torch.norm(pred[:, :, :3], dim=-1) - 1.) ** 2)

    # Ensure orthogonality between l_hat and m
    loss += wt_on_ortho * torch.mean(torch.abs(torch.sum(torch.mul(pred[:, :, :3], pred[:, :, 3:6]), dim=-1)))

    if torch.isnan(loss):
        print("target: Min: {},  Max{}".format(target.min(), target.max()))
        print("Prediction: Min: {},  Max{}".format(pred.min(), pred.max()))
        print("L2 error: {}".format(torch.mean((target - pred) ** 2)))
        print("Distance loss:{}".format(torch.mean(orientation_difference_bw_plucker_lines(target, pred) ** 2)))
        print("Orientation loss:{}".format(torch.mean(distance_bw_plucker_lines(target, pred) ** 2)))
        print("Configuration loss:{}".format(torch.mean(conf_err)))

    return loss


def articulation_lstm_loss_L2(pred, target, wt_on_ax_std=1.0, wt_on_ortho=1., extra_indiv_wts=None):
    """ L2 loss"""
    pred = pred.view(pred.size(0), -1, 8)[:, 1:, :]  # We don't need the first row as it is for single image

    err = (pred - target) ** 2
    loss = torch.mean(err)

    # Penalize spread of screw axis
    loss += wt_on_ax_std * (torch.mean(err.std(dim=1)[:6]))

    # Ensure orthogonality between l_hat and m
    loss += wt_on_ortho * torch.mean(torch.abs(torch.sum(torch.mul(pred[:, :, :3], pred[:, :, 3:6]), dim=-1)))

    if extra_indiv_wts is None:
        extra_indiv_wts = [0., 0., 0.]

    # Extra weight on axis errors 'l'
    loss += torch.mean(extra_indiv_wts[0] * err[:, :, :3])

    # Extra weight on axis errors 'm'
    loss += torch.mean(extra_indiv_wts[1] * err[:, :, 3:6])

    # Extra weight on configuration errors
    loss += torch.mean(extra_indiv_wts[2] * err[:, :, 6:])
    return loss
