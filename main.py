import argparse
import torch
import logging
import os
import numpy as np
from dataset import TextDataset
from nets import Net


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--trn-data', required=True, help='File with text corpora.', )
    parser.add_argument('--tst-data', action='append', help='File with text corpora.', )

    parser.add_argument('--start-iteration', default=0, type=int)
    parser.add_argument('--max-iterations', default=500000, type=int)
    parser.add_argument('--batch-size', default=16, type=int)
    parser.add_argument('--view-step', default=500, type=int)
    parser.add_argument('--learning-rate', default=0.0002, type=float)
    args = parser.parse_args()
    return args



#def test(iteration, model, dataset, device, max_test_lines=200):
#    model = model.eval()



def main():
    args = parse_arguments()

    trn_dataset = TextDataset(args.trn_data, min_length=20, max_length=96)
    tst_datasets = []
    if args.tst_data:
        for tst_data in args.tst_data:
            tst_datasets.append(TextDataset(tst_data, min_length=20, max_length=96,
                                            input_translator=trn_dataset.input_translator,
                                            target_translator=trn_dataset.target_translator))

    trn_loader = torch.utils.data.DataLoader(
        trn_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)

    model = Net(len(trn_dataset.input_translator.chars), len(trn_dataset.target_translator.chars), inner_channels=64)

    if args.start_iteration:
        checkpoint_path = f"checkpoint_{args.start_iteration:06d}.pth"
        logging.info(f"Restore {checkpoint_path}")
        model.load_state_dict(torch.load(checkpoint_path))

    logging.info('Start')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logging.info(f'DEVICE device')
    model = model.to(device)
    criterion = torch.nn.CrossEntropyLoss(reduction='mean')
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    trn_loss_acc = 0
    iteration = args.start_iteration
    while True:
        for batch_data, batch_labels in trn_loader:
            iteration += 1
            batch_data = batch_data.to(device).long()
            batch_labels = batch_labels.to(device).long()

            optimizer.zero_grad()
            probs = model(batch_data)
            trn_loss = criterion(probs, batch_labels)
            trn_loss.backward()
            optimizer.step()
            trn_loss = trn_loss.item()
            trn_loss_acc += trn_loss

            if iteration % args.view_step == (args.view_step - 1):
                print(f"ITERATION {iteration}")
                checkpoint_path = "checkpoint_{:06d}.pth".format(iteration + 1)
                torch.save(model.state_dict(), checkpoint_path)
                trn_loss_acc /= args.view_step
                print(f"TRAIN {iteration} loss:{trn_loss_acc:.3f}")

                #for dataset in [trn_dataset] + tst_datasets:
                #    test(iteration, model, dataset, device, max_test_lines=1000)

                trn_loss_acc = 0


if __name__ == '__main__':
    main()

