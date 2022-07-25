import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data

from tqdm import tqdm

from utils import load_data, load_adj, split_data
from models.graphcnn import GraphCNN

torch.backends.cudnn.enabled = False


def train(model, device, train_loader, optimizer, epoch):
    model.train()

    pbar = tqdm(train_loader, unit='batch')

    loss_all = 0
    num_all = 0
    for step, (batch_feature, target) in enumerate(pbar):
        batch_feature, target = batch_feature.to(device), target.to(device)
        output = model(batch_feature)
        criterion = nn.MSELoss(reduction='sum')

        # compute loss
        loss = criterion(output, target)

        # backprop
        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        loss_all += loss.detach().cpu().numpy()
        num_all += batch_feature.shape[0]

        # report
        pbar.set_description('epoch: %d' % epoch)

    train_loss = loss_all / num_all
    print("loss training: %f" % train_loss)

    return train_loss


def evaluate(model, device, loader):
    model.eval()

    mae_all = 0
    cor_all = np.zeros((2, 2))
    num_all = 0

    with torch.no_grad():
        for batch_feature, target in loader:
            batch_feature, target = batch_feature.to(device), target.to(device)
            output = model(batch_feature)
            mae_all += nn.L1Loss(reduction='sum')(output, target).detach().cpu().item()
            for i in range(output.shape[0]):
                cor_all += np.corrcoef(output[i].detach().cpu().numpy(), target[i].detach().cpu().numpy())
            num_all += batch_feature.shape[0]

    mae = mae_all / num_all
    cor = cor_all[0, 1] / num_all

    return mae, cor


def main():
    # Parameters settings
    parser = argparse.ArgumentParser(description='PyTorch implementation of graph emotion decoding (GED)')
    parser.add_argument('--subject_id', type=int, default=1,
                        help='identifier of subject (default: 1)')
    parser.add_argument('--num_sessions', type=int, default=5,
                        help='number of sessions (default: 5)')
    parser.add_argument('--category_file', type=str, default="category", choices=["category", "categcontinuous"],
                        help='type of emotion category scores: binary (category) or continuous (categcontinuous)')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='input batch size for training (default: 16)')
    parser.add_argument('--epochs', type=int, default=300,
                        help='maximum number of training epochs (default: 300)')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='initial learning rate (default: 0.01)')
    parser.add_argument('--seed', type=int, default=0,
                        help='random seed for splitting the dataset into 10 folds (default: 0)')
    parser.add_argument('--fold_idx', type=int, default=0,
                        help='fold index in 10-fold validation (should be less then 10)')
    parser.add_argument('--num_parts', type=int, default=4,
                        help='number of equal parts along each (x or y or z) axis (n, default: 4)')
    parser.add_argument('--num_activations', type=int, default=150,
                        help='number of active brain areas for each stimulus (l, default: 150)')
    parser.add_argument('--num_interactions', type=int, default=50,
                        help='number of connected/interactive brain areas for each emotion (m, default: 50)')
    parser.add_argument('--num_layers', type=int, default=5,
                        help='number of layers INCLUDING the input one (default: 5)')
    parser.add_argument('--num_mlp_layers', type=int, default=2,
                        help='number of layers for MLP EXCLUDING the input one (default: 2)')
    parser.add_argument('--hidden_dim', type=int, default=64,
                        help='number of hidden units (default: 64)')
    parser.add_argument('--final_dropout', type=float, default=0.5,
                        help='dropout ratio after the final layer (default: 0.5)')
    parser.add_argument('--neighbor_pooling_type', type=str, default="average", choices=["sum", "average"],
                        help='pooling for neighboring nodes: sum or average')
    args = parser.parse_args()

    # set up gpu device
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")

    sj = load_data(args.category_file, args.subject_id, args.num_sessions, args.num_parts)

    # 10-fold cross validation. Conduct an experiment on the fold specified by args.fold_idx.
    train_idx, test_idx = split_data(sj.features.shape[0], args.seed, args.fold_idx)
    train_data = Data.TensorDataset(sj.features[train_idx], sj.labels[train_idx])
    test_data = Data.TensorDataset(sj.features[test_idx], sj.labels[test_idx])

    num_classes = sj.labels.shape[1]
    num_nodes = sj.labels.shape[1] + sj.area_avgs.shape[1]

    # construct the emotion-brain bipartite graph
    edges = load_adj(sj.labels[train_idx], sj.area_avgs[train_idx], args.num_activations, args.num_interactions)
    edge_mat = torch.LongTensor(edges).transpose(0, 1)

    model = GraphCNN(args.num_layers, args.num_mlp_layers, train_data[0][0].shape[1], args.hidden_dim, num_classes,
                     args.final_dropout, num_nodes, edge_mat, args.neighbor_pooling_type, device).to(device)

    train_loader = Data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    test_loader = Data.DataLoader(test_data, batch_size=args.batch_size, shuffle=True)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

    for epoch in range(1, args.epochs + 1):
        train_loss = train(model, device, train_loader, optimizer, epoch)
        mae_train, cor_train = evaluate(model, device, train_loader)
        mae_test, cor_test = evaluate(model, device, test_loader)

        print("MAE train: %f, test: %f" % (mae_train, mae_test))
        print("correlation train: %f, test: %f" % (cor_train, cor_test))

        # with open(filename, 'a') as f:
        #     f.write("%f %f %f %f %f" % (train_loss, mae_train, mae_test, cor_train, cor_test))
        #     f.write("\n")

        scheduler.step()

        print("")


if __name__ == '__main__':
    main()
