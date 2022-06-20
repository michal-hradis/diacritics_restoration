import argparse
import torch
import logging
import os, sys
import numpy as np
from dataset import strip_accents, add_diacritics, load_translation
from nets import Net
from unicodedata import normalize


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-checkpoint', required=True, help='Model checkpoint.')
    parser.add_argument('--char-file', help='Prefix (excluding .input .output) of files with input and output characters.')
    args = parser.parse_args()
    return args


def test(iteration, model, dataset, device, max_test_lines=1000):
    model.eval()
    loader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=False, num_workers=0)
    count_good = 0.0
    count_all = 0
    space_id = dataset.input_translator.to_numpy(' ')
    line_counter = 0
    batch_counter = 0
    with torch.no_grad():
        for batch_data, batch_labels in loader:
            logits = model(batch_data.to(device).long())
            #probs = torch.nn.functional.softmax(logits, dim=1).cpu().numpy()
            predictions = torch.argmax(logits, dim=1).cpu().numpy()
            relevant = np.not_equal(batch_data.numpy(), space_id)
            correct = np.equal(batch_labels.numpy(), predictions)
            count_good += np.sum(np.logical_and(relevant, correct))
            count_all += np.sum(relevant)
            line_counter += batch_data.shape[0]

            if batch_counter < 20:
                base_string = dataset.input_translator.to_string(batch_data[0].numpy())
                diacritics = dataset.target_translator.to_string(predictions[0])
                print(add_diacritics(base_string, diacritics))
            batch_counter += 1

            if line_counter > max_test_lines:
                break

    print(f'TEST {dataset.get_name()} {iteration:06d} acc:{count_good/count_all:.4f}')
    model.train()


def main():
    args = parse_arguments()
    logging.info('Start')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logging.info(f'DEVICE {device}')

    input_translator = load_translation(args.char_file + '.input')
    target_translator = load_translation(args.char_file + '.target')
    model = Net(len(input_translator.chars), len(target_translator.chars), inner_channels=768)
    model.eval()
    logging.info(f"Restore {args.model_checkpoint}")
    model.load_state_dict(torch.load(args.model_checkpoint, map_location=device))
    model = model.to(device)

    min_length = 96
    for line in open('data.tst', 'r', encoding='UTF-8'):
        line = line.strip()
        segment = line
        segment = segment + ''.join(' ' for i in range(min_length - len(segment)))
        segment_input = strip_accents(segment)
        if segment_input != segment:
            continue
        net_input = torch.from_numpy(input_translator.to_numpy(segment_input)[np.newaxis])
        logits = model(net_input.to(device).long())
        predictions = torch.argmax(logits, dim=1).cpu().numpy()
        diacritics = target_translator.to_string(predictions[0, :len(segment_input)])
        print(line)
        print(normalize('NFC', add_diacritics(segment_input, diacritics)))


if __name__ == '__main__':
    main()

