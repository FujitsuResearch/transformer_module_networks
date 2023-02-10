# SPDX-License-Identifier: BSD-3-Clause-Clear
# Copyright (c) 2022 Fujitsu Limited. All rights reserved.
# Licensed under the BSD 3-Clause Clear License. See LICENSE for details.

import json

def main():
    base_path = path_cfgs.gqa_path + "questions/"
    full_path = base_path + "trainval_all_fully_inputs.json"
    testdev_path = base_path + "testdev_balanced_inputs.json"
    trainval_path = base_path + "trainval_balanced_inputs.json"
    subm_path = base_path + "submission_inputs.json"
    
    annos_full = json.load(open(full_path, 'r'))
    annos_subm = json.load(open(subm_path, 'r'))
    annos_trainval = json.load(open(trainval_path, 'r'))
    annos_testdev = json.load(open(testdev_path, 'r'))

    # annos = annos_full
    annos = annos_trainval + annos_testdev + annos_full + annos_subm

    print(len(annos))

    vocab = {}
    vocab["[PAD]"] = 0
    vocab["[EOS]"] = 1
    vocab["[UNK]"] = 2
    vocab["[SOS]"] = 3
    for idx, ann in enumerate(annos):
        prog = ann[3]
        for p in prog:            
            for i in range(1, len(p)):
                if p[i] is not None:
                    if p[i] not in vocab:
                        vocab[p[i]] = len(vocab)

    print(len(vocab))

    output = './full_vocab_gqa_all.json'
    with open(output, 'w') as f:
      json.dump(vocab, f)

if __name__ == "__main__":
    main()