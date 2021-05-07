import os
import sys
sys.path.append('..')

import numpy as np
import argparse
import torch
from torch import nn
from torch.optim import *
from torch.utils.data import TensorDataset, DataLoader
from lstm_ladder.lstm_ladder import LSTMLadder
from linear_ladder.linear_ladder import LINEARLadder


def evaluate_performance(ladder, test_loader, e, agg_cost_scaled, agg_supervised_cost_scaled,
                         agg_unsupervised_cost_scaled, args):
    correct = 0.
    total = 0.
    for batch_idx, (data, label) in enumerate(test_loader):
        if args.cuda:
            data = data.cuda()
        output = ladder.forward_encoders_clean_predict(data)
        if args.cuda:
            output = output.cpu().data.numpy()
        else:
            output = output.data.numpy()
        preds = np.argmax(output, axis=1)
        label = label.data.numpy()
        correct += np.sum(label == preds)
        total += label.shape[0]

    print("Epoch:", e + 1, "\t",
          "Total Cost:", "{:.4f}".format(agg_cost_scaled), "\t",
          "Supervised Cost:", "{:.4f}".format(agg_supervised_cost_scaled), "\t",
          "Unsupervised Cost:", "{:.4f}".format(agg_unsupervised_cost_scaled), "\t",
          "Validation Accuracy:", "{:.4f}".format(correct / total))
    return correct/total


def main():
    # command line arguments
    parser = argparse.ArgumentParser(description="Parser for Ladder network")
    parser.add_argument("--batch", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--noise_std", type=float, default=0.2)
    parser.add_argument("--data_dir", type=str, default="./data/te_series_data")
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--u_costs", type=str, default="1e-2, 1e-2, 1e-2, 1e-1, 1e-0")
    parser.add_argument("--cuda", type=bool, default=True)
    parser.add_argument("--model", type=str, default="lstm-lae")
    parser.add_argument("--model_dir", type=str, default="./model")
    parser.add_argument("--suffix", type=str, default="0")
    args = parser.parse_args()

    batch_size = args.batch
    epochs = args.epochs
    noise_std = args.noise_std
    seed = args.seed
    if args.cuda and not torch.cuda.is_available():
        print("WARNING: torch.cuda not available, using CPU.\n")
        args.cuda = False

    print("=====================")
    print("BATCH SIZE:", batch_size)
    print("EPOCHS:", epochs)
    print("RANDOM SEED:", args.seed)
    print("NOISE STD:", noise_std)
    print("CUDA:", args.cuda)
    print("=====================\n")

    np.random.seed(seed)
    torch.manual_seed(seed)
    if args.cuda:
        torch.cuda.manual_seed(seed)

    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

    unlabelled_train_data_filepath = os.path.join(args.data_dir, "unlabelled_train_data.npy")
    unlabelled_train_label_filepath = os.path.join(args.data_dir, "unlabelled_train_label.npy")
    labelled_train_data_filepath = os.path.join(args.data_dir, "labelled_train_data.npy")
    labelled_train_label_filepath = os.path.join(args.data_dir, "labelled_train_label.npy")
    test_data_filepath = os.path.join(args.data_dir, "test_data.npy")
    test_label_filepath = os.path.join(args.data_dir, "test_label.npy")

    unlabelled_train_data = np.load(unlabelled_train_data_filepath)
    unlabelled_train_label = np.load(unlabelled_train_label_filepath)

    labelled_train_data = np.load(labelled_train_data_filepath)
    labelled_train_label = np.load(labelled_train_label_filepath)

    unlabelled_train_data = np.concatenate([unlabelled_train_data, labelled_train_data], axis=0)
    unlabelled_train_label = np.concatenate([unlabelled_train_label, labelled_train_label], axis=0)

    test_data = np.load(test_data_filepath)
    test_label = np.load(test_label_filepath)

    if 'linear' in args.model:
        labelled_train_data = labelled_train_data.reshape(labelled_train_data.shape[0], -1)
        unlabelled_train_data = unlabelled_train_data.reshape(unlabelled_train_data.shape[0], -1)
        test_data = test_data.reshape(test_data.shape[0], -1)

    # Create DataLoaders
    unlabelled_dataset = TensorDataset(torch.FloatTensor(unlabelled_train_data),
                                       torch.LongTensor(unlabelled_train_label))
    unlabelled_loader = DataLoader(unlabelled_dataset, batch_size=batch_size, shuffle=True, **kwargs)
    test_dataset = TensorDataset(torch.FloatTensor(test_data),
                                 torch.LongTensor(test_label))
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, **kwargs)

    # Configure the Ladder
    starter_lr = 0.01
    n_times = 30
    n_classes = 16
    encoder_sizes = [40, 30, 20, 16]
    decoder_sizes = [20, 30, 40, 50]
    if 'linear' in args.model:
        encoder_sizes = [1000, 500, 100, 16]
        decoder_sizes = [100, 500, 1000, 1500]
    unsupervised_costs_lambda = [float(x) for x in args.u_costs.split(",")]
    encoder_activations = ["leakyrelu"] * 4
    encoder_train_bn_scalings = [True] * 4
    if args.model == 'lstm-lae':
        ladder = LSTMLadder(n_times, n_classes, encoder_sizes, decoder_sizes, 
                            encoder_activations, encoder_train_bn_scalings, noise_std, 
                            args.cuda)
    if args.model == 'linear-lae':
        ladder = LINEARLadder(n_classes, encoder_sizes, decoder_sizes, 
                              encoder_activations, encoder_train_bn_scalings, noise_std, 
                              args.cuda)
    # for name, param in ladder.named_parameters():
    #     if name.startswith("weight"):
    #         nn.init.xavier_normal_(param)
    #     else:
    #         nn.init.zeros_(param)

    optimizer = SGD(ladder.parameters(), lr=starter_lr)
    loss_supervised = torch.nn.CrossEntropyLoss()
    loss_unsupervised = torch.nn.MSELoss()

    if args.cuda:
        ladder.cuda()

    assert len(unsupervised_costs_lambda) == len(decoder_sizes) + 1
    assert len(encoder_sizes) == len(decoder_sizes)

    print("")
    print("========NETWORK=======")
    print(ladder)
    print("======================")

    print("")
    print("==UNSUPERVISED-COSTS==")
    print(unsupervised_costs_lambda)

    print("")
    print("=====================")
    print("TRAINING\n")

    train_losses = []
    train_supervised_losses = []
    train_unsupervised_losses = []
    test_accuracies = []
    accuracy = 0.
    for e in range(epochs):
        num_batches = 0
        ladder.train()
        ind_labelled = 0
        ind_limit = np.ceil(float(labelled_train_data.shape[0]) / batch_size)

        for batch_idx, (batch_unlabelled_train_data, _) in enumerate(unlabelled_loader):
            if ind_labelled == ind_limit:
                randomize = np.arange(labelled_train_data.shape[0])
                np.random.shuffle(randomize)
                labelled_train_data = labelled_train_data[randomize]
                labelled_train_label = labelled_train_label[randomize]
                ind_labelled = 0

            labelled_start = batch_size * ind_labelled
            labelled_end = batch_size * (ind_labelled + 1)
            ind_labelled += 1
            batch_labelled_train_data = torch.FloatTensor(labelled_train_data[labelled_start:labelled_end])
            batch_labelled_train_label = torch.LongTensor(labelled_train_label[labelled_start:labelled_end])

            if args.cuda:
                batch_labelled_train_data = batch_labelled_train_data.cuda()
                batch_labelled_train_label = batch_labelled_train_label.cuda()
                batch_unlabelled_train_data = batch_unlabelled_train_data.cuda()

            # LABELLED CLASSIFICATION
            output_noise_predict = ladder.forward_encoders_noise_predict(batch_labelled_train_data)
            cost_supervised_classification = loss_supervised(output_noise_predict, batch_labelled_train_label)
            optimizer.zero_grad()
            cost_supervised_classification.backward()
            optimizer.step()

            # UNLABELLED RECONSTRUCTION
            # do a noisy pass for unlabelled_data
            output_noise_unlabelled = ladder.forward_encoders_noise(batch_unlabelled_train_data)
            tilde_z_layers_unlabelled = ladder.get_encoders_tilde_z(reverse=True)
            tilde_z_bottom_unlabelled = ladder.get_encoder_tilde_z_bottom()

            # do a clean pass for unlabelled data
            _ = ladder.forward_encoders_clean(batch_unlabelled_train_data)
            z_layers_unlabelled = ladder.get_encoders_z(reverse=True)

            # pass through decoders
            hat_z_layers_unlabelled = ladder.forward_decoders(tilde_z_layers_unlabelled,
                                                              output_noise_unlabelled,
                                                              tilde_z_bottom_unlabelled)

            # BP
            cost_unsupervised = 0.
            assert len(z_layers_unlabelled) == len(hat_z_layers_unlabelled)
            for cost_lambda, z, bn_hat_z in zip(unsupervised_costs_lambda, z_layers_unlabelled, hat_z_layers_unlabelled):
                c = cost_lambda * loss_unsupervised(bn_hat_z, z)
                cost_unsupervised += c
            optimizer.zero_grad()
            cost_unsupervised.backward()
            optimizer.step()
           
            cost_supervised = cost_supervised_classification
            cost = cost_supervised + cost_unsupervised

            cost_value = cost.item()
            supervised_cost_value = cost_supervised.item()
            unsupervised_cost_value = cost_unsupervised.item()
            num_batches += 1

        train_losses.append(cost_value)
        train_supervised_losses.append(supervised_cost_value)
        train_unsupervised_losses.append(unsupervised_cost_value)
        # Evaluation
        ladder.eval()
        test_accuracy = evaluate_performance(ladder, test_loader, e,
                                             cost_value,
                                             supervised_cost_value,
                                             unsupervised_cost_value,
                                             args)
        if test_accuracy > accuracy:
            accuracy = test_accuracy
            modelpath = os.path.join(args.model_dir, "%s%s.pkl" % (args.model, args.suffix))
            torch.save(ladder.state_dict(), modelpath)
        test_accuracies.append(test_accuracy)
        ladder.train()
                    
    print("=====================\n")
    print("Total Aggregated Mean Loss: ", train_losses)
    print("Unsupervised Aggregated Mean Loss: ", train_unsupervised_losses)
    print("Supervised Aggregated Mean Loss: ", train_supervised_losses)
    print("Test Accuracy: ", test_accuracies)

    print("Done :)")


if __name__ == "__main__":
    main()
