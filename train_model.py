import argparse

import numpy as np
import torch
from ScrewNet.dataset import ArticulationDataset, RigidTransformDataset
from ScrewNet.model_trainer import ModelTrainer
from ScrewNet.models import ScrewNet, ScrewNet_2imgs, ScrewNet_NoLSTM
from ScrewNet.loss import articulation_lstm_loss_spatial_distance, articulation_lstm_loss_spatial_distance_RT, \
    articulation_lstm_loss_L2
from ScrewNet.noise_models import DropPixels

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train object learner on articulated object dataset.")
    parser.add_argument('--name', type=str, help='jobname', default='test')
    parser.add_argument('--train-dir', type=str, default='../data/test/microwave/')
    parser.add_argument('--test-dir', type=str, default='../data/test/microwave/')
    parser.add_argument('--ntrain', type=int, default=1,
                        help='number of total training samples (n_object_instants)')
    parser.add_argument('--ntest', type=int, default=1, help='number of test samples (n_object_instants)')
    parser.add_argument('--epochs', type=int, default=10, help='number of iterations through data')
    parser.add_argument('--batch', type=int, default=128, help='batch size')
    parser.add_argument('--nwork', type=int, default=8, help='num_workers')
    parser.add_argument('--val-freq', type=int, default=5, help='frequency at which to validate')
    parser.add_argument('--cuda', action='store_true', default=False, help='use cuda')
    parser.add_argument('--learning-rate', type=float, default=1e-3)
    parser.add_argument('--device', type=int, default=0, help='cuda device')
    parser.add_argument('--model-type', type=str, default='lstm', help='screw, noLSTM, 2imgs, l2')
    parser.add_argument('--load-wts', action='store_true', default=False, help='Should load model wts from prior run?')
    parser.add_argument('--wts-dir', type=str, default='models/', help='Dir of saved model wts')
    parser.add_argument('--prior-wts', type=str, default='test', help='Name of saved model wts')
    parser.add_argument('--fix-seed', action='store_true', default=False, help='Should fix seed or not')

    args = parser.parse_args()

    print(args)
    print('cuda?', torch.cuda.is_available())

    if args.fix_seed:
        torch.manual_seed(1)
        np.random.seed(1)

    noiser = DropPixels(p=0.1)

    if args.model_type == '2imgs':
        '''Rigid Transform Datasets'''
        trainset = RigidTransformDataset(args.args.ntrain,
                                         args.train_dir,
                                         transform=noiser)

        testset = RigidTransformDataset(args.ntest,
                                        args.test_dir,
                                        transform=noiser)

        loss_fn = articulation_lstm_loss_spatial_distance_RT

        network = ScrewNet_2imgs(n_output=8)

    elif args.model_type == 'noLSTM':
        trainset = ArticulationDataset(args.ntrain,
                                       args.train_dir,
                                       transform=noiser)

        testset = ArticulationDataset(args.ntest,
                                      args.test_dir,
                                      transform=noiser)

        loss_fn = articulation_lstm_loss_spatial_distance
        network = ScrewNet_NoLSTM(seq_len=16, fc_replace_lstm_dim=1000, n_output=8)

    elif args.model_type == 'l2':
        trainset = ArticulationDataset(args.ntrain,
                                       args.train_dir,
                                       transform=noiser)

        testset = ArticulationDataset(args.ntest,
                                      args.test_dir,
                                      transform=noiser)

        loss_fn = articulation_lstm_loss_L2
        network = ScrewNet(lstm_hidden_dim=1000, n_lstm_hidden_layers=1, n_output=8)

    else:  # Default: 'screw'
        trainset = ArticulationDataset(args.ntrain,
                                       args.train_dir,
                                       transform=noiser)

        testset = ArticulationDataset(args.ntest,
                                      args.test_dir,
                                      transform=noiser)

        loss_fn = articulation_lstm_loss_spatial_distance
        network = ScrewNet(lstm_hidden_dim=1000, n_lstm_hidden_layers=1, n_output=8)

    testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch,
                                             shuffle=True, num_workers=args.nwork,
                                             pin_memory=True)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch,
                                              shuffle=True, num_workers=args.nwork,
                                              pin_memory=True)

    # Load Saved wts
    if args.load_wts:
        network.load_state_dict(torch.load(args.wts_dir + args.prior_wts + '.net'))

    # setup trainer
    if torch.cuda.is_available():
        device = torch.device(args.device)
    else:
        device = torch.device('cpu')

    optimizer = torch.optim.Adam(network.parameters(),
                                 lr=args.learning_rate,
                                 weight_decay=1e-2)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    # ## Debug
    # torch.autograd.set_detect_anomaly(True)

    trainer = ModelTrainer(model=network,
                           train_loader=trainloader,
                           test_loader=testloader,
                           optimizer=optimizer,
                           scheduler=scheduler,
                           criterion=loss_fn,
                           epochs=args.epochs,
                           name=args.name,
                           test_freq=args.test_freq,
                           device=args.device,
                           ndof=args.ndof)

    # train
    best_model = trainer.train()

    # #Test best model
    # trainer.test_best_model(best_model, fname_suffix='_posttraining')
