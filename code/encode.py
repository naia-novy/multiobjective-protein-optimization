""" encoding protein variants with one-hot, etc """
from typing import List

import numpy as np
import torch
import torch.nn as nn
from torchtext import vocab
from collections import OrderedDict
import random

import utils

# all the encoding characters we might encounter
AA_CHARS = ["A", "C", "D", "E", "F", "G", "H", "I", "K", "L",
            "M", "N", "P", "Q", "R", "S", "T", "V", "W", "Y"]

NUC_CHARS = ['A','T','C','G']

CODONS = {
    'I': ['ATA', 'ATC', 'ATT'],
    'M': ['ATG'],
    'T': ['ACA', 'ACC', 'ACG', 'ACT'],
    'N': ['AAC', 'AAT'],
    'K': ['AAA', 'AAG'],
    'S': ['AGC', 'AGT', 'TCA', 'TCC', 'TCG', 'TCT'],
    'R': ['AGA', 'AGG', 'CGA', 'CGC', 'CGG', 'CGT'],
    'L': ['CTA', 'CTC', 'CTG', 'CTT', 'TTA', 'TTG'],
    'P': ['CCA', 'CCC', 'CCG', 'CCT'],
    'H': ['CAC', 'CAT'],
    'Q': ['CAA', 'CAG'],
    'V': ['GTA', 'GTC', 'GTG', 'GTT'],
    'A': ['GCA', 'GCC', 'GCG', 'GCT'],
    'D': ['GAC', 'GAT'],
    'E': ['GAA', 'GAG'],
    'G': ['GGA', 'GGC', 'GGG', 'GGT'],
    'F': ['TTC', 'TTT'],
    'Y': ['TAC', 'TAT'],
    '*': ['TAA', 'TAG', 'TGA'],
    'C': ['TGC', 'TGT'],
    'W': ['TGG']
    }

CODON_CHARS = [x for xs in list(CODONS.values()) for x in xs]

# mapping from amino acid characters --> integer
AA_MAP = {c: i for i, c in enumerate(AA_CHARS)}
NUC_MAP = {c: i for i, c in enumerate(NUC_CHARS)}
CODON_MAP = {c: i for i, c in enumerate(CODON_CHARS)}

# set up aa2ind, which is used for aaindex encoding
AAs = 'ACDEFGHIKLMNPQRSTVWY*'
aa2ind = vocab.vocab(OrderedDict([(a, 1) for a in AAs]))
aa2ind.set_default_index(20)  # set unknown characters to gap

def reverse_translate(variants: list, augmentation_factor: int, encoding: str):
    """ Convert a protein sequence into a DNA sequence (used for nucleotide augmentation encoding)"""

    DNA_seqs = []
    sequence_variant = [] # To deal with augmentation
    for variant in variants:
        DNA_seq = []
        for AA in variant:
            DNA_seq.append(random.choice(CODONS[AA]))

        if encoding == 'one_hot_nuc_uni' or encoding == 'three_hot_nuc_uni':
            DNA_seq = list(''.join(DNA_seq))

        sequence_variant.append(DNA_seq)
    DNA_seqs.append(sequence_variant)
    DNA_seqs = np.array(DNA_seqs)

    new_shape = list(DNA_seqs.shape)
    new_shape[0] *= augmentation_factor
    new_shape = tuple(new_shape)

    DNA_seqs = np.repeat(DNA_seqs[np.newaxis], augmentation_factor, axis=0).reshape(new_shape)

    return DNA_seqs.tolist()


def encode_int_seqs(variants, wt_aa, ds_name, frame_shift):
    """ encode variants as a sequence of integers where each integer corresponds to an amino acid """

    # Convert wild type seq to integer encoding
    wt_int = np.zeros(len(wt_aa), dtype=np.uint8)

    # Tile the wild-type seq
    seq_ints = np.tile(wt_int, (len(variants), 1))

    if len(variants[0]) == len(wt_aa): # provided as sequence format
        for i, variant in enumerate(variants):
            # Special handling if we want to encode the wild-type seq
            # the seq_ints array is already filled with WT, so all we have to do is just ignore it
            if variant == wt_aa:
                continue

            for pos_idx in range(len(variant)):
                if variant[pos_idx] != wt_aa[pos_idx]:
                    replacement = AA_MAP[variant[pos_idx]]
                    seq_ints[i, pos_idx] = replacement

    else: # Variants are a list of mutations [mutation1, mutation2, ....]
        for i, variant in enumerate(variants):
            # Special handling if we want to encode the wild-type seq
            # the seq_ints array is already filled with WT, so all we have to do is just ignore it
            if variant == "_wt" or variant == '':
                continue

            variant = variant.split(',')

            for mutation in variant:
                mutation = mutation.strip() # Removes whitespace if present
                # Mutations are in the form <original char><position><replacement char>
                position = int(mutation[1:-1])
                position -= frame_shift
                replacement = AA_MAP[mutation[-1]]

                seq_ints[i, position] = replacement

    return seq_ints.astype(int)


def mut2seq(variants, wt_aa, frame_shift):
    """ Convert a mutant variant to a list of AA representing the protein sequence """

    # Initialize wt_list
    wt_list = list(wt_aa)

    seq_list = []
    for i, variant in enumerate(variants):
        # Special handling if we want to encode the wild-type seq
        # the seq_ints array is already filled with WT, so all we have to do is just ignore it
        if variant == "_wt":
            seq_list.append(wt_list)
            # variants are a list of mutations [mutation1, mutation2, ....]

        variant = variant.split(',')
        var_seq = wt_list.copy()

        for mutation in variant:
            mutation = mutation.strip()  # Removes whitespace if present
            # mutations are in the form <original char><position><replacement char>
            position = int(mutation[1:-1])
            position -= frame_shift
            var_seq[position] = mutation[-1]

        seq_list.append(var_seq)

    return seq_list


def seq2mut(sequences, wt_aa):
    """ Convert a list of sequence strings to a list of mutant variants """
    variants = []
    for seq in sequences:
        mutations = []
        for i,aa in enumerate(seq):
            if aa != wt_aa[i]:
                mutation = f'{wt_aa[i]}{i}{aa}'
                mutations.append(mutation)
        variants.append(','.join(mutations))

    return variants


def seq2ind(sequences,CHARS=AA_CHARS):
    """ Take a list of mutant protein sequences and return an indexed version (will later serve as input to embed) """

    aa2ind = vocab.vocab(OrderedDict([(a, 1) for a in CHARS]))
    # This should not be needed if inputs do not have * or -
    # aa2ind.set_default_index(len(CHARS)-1)  # set unknown characters to gap

    idxed_seqs = []
    for aug_seq in sequences:
        idxed_sequence_variants = []
        for seq in aug_seq:
            idxed_seq = aa2ind(list(seq))
            idxed_sequence_variants.append(idxed_seq)
        idxed_seqs.append(idxed_sequence_variants)

    return np.array(idxed_seqs).astype(np.int32)

def encode_one_hot(int_seqs, CHARS):
    """ Encode given integer sequences as one-hot encoding """
    # If you plan on saving this encoded data to disk, note that it is dtype float32
    # to save disk space, you can encode it as a boolean type instead...
    one_hot = np.eye(len(CHARS))[int_seqs]
    return one_hot.astype(np.float32)

def encode(encoding: str, variants: List[str], ds_name: str,
           frame_shift: int, ncomp: int, augmentation_factor: int, aaindex = None, batch_converter=None):
    """ Wrapper funciton for all encoding options """

    # Load in WT sequence
    ds_info = utils.load_ds_index()[ds_name]
    wt_aa = ds_info["wt_aa"]

    # Different encoding methods: name in args file must match one of these options
    if encoding == "one_hot_AA":

        if len(variants[0]) == len(wt_aa):
            variants = seq2mut(variants, wt_aa)
            frame_shift = 0
        int_seqs = encode_int_seqs(variants, wt_aa, ds_name, frame_shift)
        return encode_one_hot(int_seqs,AA_CHARS)
    elif encoding == "one_hot_nuc_uni":
        if len(variants[0]) != len(wt_aa):
            variants = mut2seq(variants, wt_aa, frame_shift)
        seqs = reverse_translate(variants,augmentation_factor=augmentation_factor,encoding=encoding)
        seqs = seq2ind(seqs,NUC_CHARS)
        return encode_one_hot(seqs,NUC_CHARS)
    elif encoding == "three_hot_nuc_uni":
        if len(variants[0]) != len(wt_aa):
            variants = mut2seq(variants, wt_aa, frame_shift)
        seqs = reverse_translate(variants,augmentation_factor=augmentation_factor,encoding=encoding)
        seqs = seq2ind(seqs,NUC_CHARS)
        seqs = encode_one_hot(seqs,NUC_CHARS)
        return np.reshape(seqs, (len(seqs[0]),-1,12))
    elif encoding == "one_hot_nuc_tri":
        if len(variants[0]) != len(wt_aa):
            variants = mut2seq(variants, wt_aa, frame_shift)
        seqs = reverse_translate(variants,augmentation_factor=augmentation_factor,encoding=encoding)
        seqs = seq2ind(seqs,CODON_CHARS)
        return encode_one_hot(seqs,CODON_CHARS)
    # one_hot_aaindex concatenates the one-hot encoding and aaindex encoding together
    elif encoding == 'one_hot_aaindex':
        # set up aaindex encoding
        # Note that when aaindex and one hot are combined, parameters for aaindex conversion are not allowed to change
        aaindex = np.array(
            [[float(f) for f in l.split(',')[1:]] for l in open('data/pca-19_raw.csv').read().strip().split('\n')[1:]])
        aaindex = (aaindex - aaindex.mean(0)) / aaindex.std(0)  # standardize
        aaindex = np.vstack([aaindex, np.zeros((1, 19))])  # add final row to include gap -
        aaindex = torch.from_numpy(aaindex).float()
        aaindex = aaindex[:, :ncomp]
        embed = nn.Embedding.from_pretrained(aaindex, freeze=True)

        seqs = variants.copy()
        if len(variants[0]) != len(wt_aa): # If variants is mutations instead of sequences, this will be True
            seqs = mut2seq(seqs,wt_aa,frame_shift)
        aaindex_seqs = seq2ind(seqs, AA_CHARS)
        aaindex_seqs = embed(torch.tensor(aaindex_seqs))
        aaindex_seqs = aaindex_seqs.view(-1, len(wt_aa), ncomp)

        # one hot encoding
        int_seqs = encode_int_seqs(variants, wt_aa, ds_name, frame_shift)
        one_hot_seqs = encode_one_hot(int_seqs,AA_CHARS)

        # concatenate two types
        combined_enc = np.concatenate((one_hot_seqs,aaindex_seqs),axis=2)

        return combined_enc
    elif encoding == 'aaindex':
        # this indicates that sequence data was given instead I added this for testing random
        # sequence incorperation for which I only saved sequences, not mutations
        if len(variants[0]) != len(wt_aa):
            variants = mut2seq(variants,wt_aa,frame_shift)
        seqs = seq2ind(variants, AA_CHARS)

        return seqs
    elif 'ESM_' in encoding:
        seqs = batch_converter([(v,v) for v in variants])[2]
        return seqs
    elif 'METL-G-' in encoding:
        if len(variants[0]) != len(wt_aa):
            # assume mutants were provided
            variants = batch_converter.encode_variants(wt_aa, variants)
        seqs = batch_converter.encode_sequences(variants)
        return seqs

    else:
        raise ValueError("Encoding '{}' not implemented!".format(wt_aa))


class EncodeAAindex:
    """ Class for aaindex encoding :"""
    def __init__(self, ncomp=6, aaindex = None):
        # Set up converter for aa to index
        self.aa2ind = aa2ind

        if aaindex == None:
            aaindex = np.array(
                [[float(f) for f in l.split(',')[1:]] for l in open('data/pca-19_raw.csv').read().strip().split('\n')[1:]])

            aaindex = (aaindex - aaindex.mean(0)) / aaindex.std(0)  # standardize
            aaindex = np.vstack([aaindex, np.zeros((1, 19))])  # add final row to include gap -
            aaindex = torch.from_numpy(aaindex).float()
            self.aaindex = aaindex[:, :ncomp]
        else:
            self.aaindex = aaindex
    #
    # def mut2seq(self,variants,wt_aa,ds_name, frame_shift):
    #     # Make sequence string and mutate residues
    #     wt_list = list(wt_aa)
    #
    #     seq_list = []
    #     for i, variant in enumerate(variants):
    #         # special handling if we want to encode the wild-type seq
    #         # the seq_ints array is already filled with WT, so all we have to do is just ignore it
    #         if variant == "_wt":
    #             continue
    #             # variants are a list of mutations [mutation1, mutation2, ....]
    #
    #         variant = variant.split(',')
    #         var_seq = wt_list.copy()
    #
    #         for mutation in variant:
    #             mutation = mutation.strip()  # Removes whitespace if present
    #             # mutations are in the form <original char><position><replacement char>
    #             position = int(mutation[1:-1])
    #             position -= frame_shift
    #             var_seq[position] = mutation[-1]
    #
    #         seq_list.append(var_seq)
    #
    #     return seq_list
    #
    # def seq2ind(self,sequences):
    #     # Take a list of mutant protein sequences and return an indexed version (will later serve as input to embed)
    #     idxed_seqs = []
    #     for seq in sequences:
    #         idxed_seq = self.aa2ind(list(seq))
    #         idxed_seqs.append(idxed_seq)
    #
    #     return np.array(idxed_seqs).astype(np.int32)
    #
