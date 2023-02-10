# SPDX-License-Identifier: BSD-3-Clause-Clear
# Copyright (c) 2022 Fujitsu Limited. All rights reserved.
# Licensed under the BSD 3-Clause Clear License. See LICENSE for details.

import os

class PATH:
    def __init__(self):

        self.root_path = "./modular_transformer/"

        self.results_path = "./"
        self.storage_path = self.results_path + "transformer_clevr/"

        self.clevr_path = ''
        self.gqa_path = ''
        self.closure_path = ''
        self.bert_model = 'bert-base-uncased-vocab.txt'

        self.save_path = self.storage_path + ''

        self.config_file = self.root_path + 'config/bert_base_6layer_6conect.json'

        self.vocab_path = self.root_path + 'datasets/vocab.json'
        self.func_vocab_path = self.root_path + 'datasets/func_vocab.json'
        self.args_vocab_path = self.root_path + 'datasets/args_vocab.json'

        self.path_dict_corpus_train = {}
        self.path_dict_corpus_val = {}
        self.path_dict_annotation_train = {}
        self.path_dict_annotation_val = {}
        self.path_dict_proposal_train = {}
        self.path_dict_proposal_val = {}

        self.path_dict_corpus_train['clevr'] = self.clevr_path + 'extracted_features/train/feature/'
        self.path_dict_corpus_train['clevr_raw'] = self.clevr_path + 'CLEVR_v1.0/images/train/'
        self.path_dict_corpus_train['cgt'] = self.clevr_path + 'extracted_features/cgt/train/feature/'
        self.path_dict_corpus_train['cgt_raw'] = self.clevr_path + 'CLEVR_CoGenT_v1.0/images/trainA/'
        self.path_dict_corpus_train['gqa'] = self.gqa_path + 'vg_gqa_imgfeat/vg_gqa_obj36.tsv'

        self.path_dict_corpus_val['clevr'] = self.clevr_path + 'extracted_features/val/feature/'
        self.path_dict_corpus_val['clevr_raw'] = self.clevr_path + 'CLEVR_v1.0/images/val/'
        self.path_dict_corpus_val['cgt'] = self.clevr_path + 'extracted_features/cgt/val_b/feature/'
        self.path_dict_corpus_val['cgt_raw'] = self.clevr_path + 'CLEVR_CoGenT_v1.0/images/valB/'
        self.path_dict_corpus_val['valA'] = self.clevr_path + 'CLEVR_CoGenT_v1.0/images/valA/'
        self.path_dict_corpus_val['valB'] = self.clevr_path + 'CLEVR_CoGenT_v1.0/images/valB/'
        self.path_dict_corpus_val['val'] = self.clevr_path + 'CLEVR_v1.0/images/val/'
        self.path_dict_corpus_val['valA_obj'] = self.clevr_path + 'extracted_features/cgt/val_a/feature/'
        self.path_dict_corpus_val['valB_obj'] = self.clevr_path + 'extracted_features/cgt/val_b/feature/'
        self.path_dict_corpus_val['val_obj'] = self.clevr_path + 'extracted_features/val/feature/'


        self.path_dict_annotation_train['clevr'] = self.clevr_path + 'CLEVR_v1.0/questions/CLEVR_train_questions.json'
        self.path_dict_annotation_train['cgt'] =  self.clevr_path + 'CLEVR_CoGenT_v1.0/questions/CLEVR_trainA_questions.json'
        self.path_dict_annotation_train['gqa'] = self.gqa_path + "trainval.json"
        self.path_dict_annotation_train['gqa_prog'] = self.gqa_path + 'questions/trainval_balanced_inputs.json'

        self.path_dict_annotation_val['clevr'] = self.clevr_path + 'CLEVR_v1.0/questions/CLEVR_val_questions.json'
        self.path_dict_annotation_val['cgt'] = self.clevr_path + 'CLEVR_CoGenT_v1.0/questions/CLEVR_valB_questions.json'
        self.path_dict_annotation_val['valA'] = self.clevr_path + 'CLEVR_CoGenT_v1.0/questions/CLEVR_valA_questions.json'
        self.path_dict_annotation_val['valB'] = self.clevr_path + 'CLEVR_CoGenT_v1.0/questions/CLEVR_valB_questions.json'
        self.path_dict_annotation_val['val'] = self.clevr_path + 'CLEVR_v1.0/questions/CLEVR_val_questions.json'

        self.path_dict_corpus_val['gqa'] = self.gqa_path + 'vg_gqa_imgfeat/gqa_testdev_obj36.tsv'
        self.path_dict_annotation_val['gqa'] = self.gqa_path + 'testdev.json'
        self.path_dict_annotation_val['gqa_prog'] = self.gqa_path + 'questions/testdev_balanced_inputs.json'

        self.path_dict_proposal_train['cgt'] = self.clevr_path + 'mask_rcnn/results/cgt_train/detections.pkl'
        self.path_dict_proposal_val['valA'] = self.clevr_path + 'mask_rcnn/results/cgt_val_/detections.pkl'
        self.path_dict_proposal_val['valB'] = self.clevr_path + 'mask_rcnn/results/cgt_val_/detections.pkl'
        self.path_dict_proposal_val['val'] = None