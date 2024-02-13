import pandas as pd
import numpy as np
import math
from copy import deepcopy
from kipoi.data import Dataset, kipoi_dataloader
from kipoi_conda.dependencies import Dependencies
from kipoi.specs import Author
from kipoi_utils.utils import default_kwargs
from kipoiseq.transforms import ReorderedOneHot


deps = Dependencies(conda=['numpy', 'pandas', 'kipoiseq'])

package_authors = [Author(name='Max Schubach', github='visze')]

# Object exported on import *
__all__ = ['StringDreamChallange1D', 'SeqDreamChallange1D']


@kipoi_dataloader(override={"dependencies": deps, 'info.authors': package_authors})
class StringDreamChallange1D(Dataset):
    """
    info:
        doc: >
            Dataloader for a tab-delimited input file containing in the first column the sequence and the second the class label.
            Sequences converts them into one-hot encoded sequence and to an encoded.
            Returned sequences are of the type np.array with the shape inferred from the arguments: `alphabet_axis`
            and `dummy_axis`.
    args:
        tsv_file:
            doc: TODO
            example:
              url: TODO
              md5: TODO
        trim_adapters:
            doc: False, removes 17 bp from thebeginning and 13 from the end.
        sequence_length:
            doc: 110, required sequence length.
        label_dtype:
            doc: 'specific data type for labels, Example: `float` or `np.float32`'
        label_start_column:
            doc: 'specific start column of labels. default 1 (second column)'
        fit_sequence:
            doc: 'fit the sequence to the given length by removing or extending nucleotides on both sides'
        label_stop_column:
            doc: 'specific end column of labels. default is none(fourth column)'
        force_upper:
            doc: Force uppercase output of sequences
        ignore_targets:
            doc: if True, don't return any target variables
    output_schema:
        inputs:
            name: seq
            shape: ()
            doc: DNA sequence as string
            special_type: DNAStringSeq
        targets:
            shape: (None,)
            doc: (optional) values following the bed-entries
    """

    def __init__(self,
                 tsv_file,
                 trim_adapters=False,
                 fit_sequence=False,
                 sequence_length=110,
                 label_start_column=1,
                 label_stop_column=None,
                 force_upper=True,
                 label_dtype=None,
                 ignore_targets=False):

        self.tsv_file = tsv_file
        self.label_dtype = label_dtype
        self.sequence_length = sequence_length
        self.label_start_column = label_start_column
        self.label_stop_column = label_stop_column
        self.trim_adapters = trim_adapters
        self.fit_sequence = fit_sequence
        self.force_upper = force_upper
        self.ignore_targets = ignore_targets

        self.vector_left = "ctatgcggtgtgaaataccgcacagatgcgtaaggagaaaataccgcatcaggaaattgtaagcgttaatattttgttaaaattcgcgttaaatttttgttaaatcagctcattttttaaccaataggccgaaatcggcaaaatcccttataaatcaaaagaatagaccgagatagggttgagtgttgttccagtttggaacaagagtccactattaaagaacgtggactccaacgtcaaagggcgaaaaaccgtctatcagggcgatggcccactacgtgaaccatcaccctaatcaagtGCTAGCAGGAATGATGCAAAAGGTTCCCGATTCGAAC".upper()
        self.vector_right = "CTTAATTAAAAAAAGATAGAAAACATTAGGAGTGTAACACAAGACTTTCGGATCCTGAGCAGGCAAGATAAACGAAGGCAAAGatgtctaaaggtgaagaattattcactggtgttgtcccaattttggttgaattagatggtgatgttaatggtcacaaattttctgtctccggtgaaggtgaaggtgatgctacttacggtaaattgaccttaaaattgatttgtactactggtaaattgccagttccatggccaaccttagtcactactttaggttatggtttgcaatgttttgctagatacccagatcatatgaaacaacatgactttttcaagtctgccatgccagaaggttatgttcaagaaagaactatttttttcaaagatgacggtaactacaagaccagagctgaagtcaagtttgaaggtgataccttagttaatagaatcgaattaaaaggtattgattttaaagaagatggtaacattttaggtcacaaattggaatacaactataactctcacaatgtttacatcactgctgacaaacaaaagaatggtatcaaagctaacttcaaaattagacacaacattgaagatggtggtgttcaat".upper()

        self.df = pd.read_csv(self.tsv_file,header=None, sep='\t')
        if (not self.fit_sequence):
            self.df = self.df[self.df.iloc[:,0].apply(lambda x: len(str(x))) == self.sequence_length]

    def __len__(self):
        return self.df.shape[0]


    def __getitem__(self, idx):

        row = self.df.iloc[idx]
    
        seq = row.iloc[0]
        if self.trim_adapters:
            seq = seq[17:self.sequence_length-13]
        if self.fit_sequence:
            seq_len = len(seq)
            missing = self.sequence_length - seq_len
            left = math.ceil(abs(missing)/2)
            
            if missing < 0:
                seq = seq[left:self.sequence_length+left]
            elif missing > 0:
                right = math.floor(abs(missing)/2)
                seq = self.vector_left[len(self.vector_left)-left:] + seq + self.vector_right[:right]

        if self.ignore_targets:
            labels = {}
        else:
            if self.label_stop_column == None:
                labels = row.iloc[self.label_start_column:].values.astype(self.label_dtype)
            else:
                labels = row.iloc[self.label_start_column:self.label_stop_column].values.astype(self.label_dtype)
            
        return {
            "inputs": np.array(seq),
            "targets": labels,
        }
    
    def get_targets(self):
        if self.label_stop_column == None:
            return self.df.iloc[:, self.label_start_column:].values.astype(self.label_dtype)
        else:
            return self.df.iloc[:, self.label_start_column:self.label_stop_column].values.astype(self.label_dtype)
    @classmethod
    def get_output_schema(cls):
        output_schema = deepcopy(cls.output_schema)
        kwargs = default_kwargs(cls)
        ignore_targets = kwargs['ignore_targets']
        if ignore_targets:
            output_schema.targets = None
        return output_schema



@kipoi_dataloader(override={"dependencies": deps, 'info.authors': package_authors})
class SeqDreamChallange1D(Dataset):
    """
    info:
        doc: >
            Dataloader for a combination of fasta and tab-delimited input files such as bed files. The dataloader extracts
            regions from the fasta file as defined in the tab-delimited `intervals_file` and converts them into one-hot encoded
            format. Returned sequences are of the type np.array with the shape inferred from the arguments: `alphabet_axis`
            and `dummy_axis`.
    args:
        tsv_file:
            doc: TODO
            example:
              url: TODO
              md5: TODO
        trim_adapters:
            doc: False, removes 17 bp from thebeginning and 13 from the end.
        sequence_length:
            doc: 110, required sequence length.
        label_dtype:
            doc: 'specific data type for labels, Example: `float` or `np.float32`'
        label_start_column:
            doc: 'specific start column of lables. default 1 (second column)'
        label_stop_column:
            doc: 'specific end column of labels. default is none(fourth column)'
        fit_sequence:
            doc: 'fit the sequence to the given length by removing or extending nucleotides on both sides'  
        alphabet_axis:
            doc: axis along which the alphabet runs (e.g. A,C,G,T for DNA)
        dummy_axis:
            doc: defines in which dimension a dummy axis should be added. None if no dummy axis is required.
        alphabet:
            doc: >
                alphabet to use for the one-hot encoding. This defines the order of the one-hot encoding.
                Can either be a list or a string: 'ACGT' or ['A, 'C', 'G', 'T']. Default: 'ACGT'
        dtype:
            doc: 'defines the numpy dtype of the returned array. Example: int, np.int32, np.float32, float'
        ignore_targets:
            doc: if True, don't return any target variables
    output_schema:
        inputs:
            name: seq
            shape: (None, 4)
            doc: One-hot encoded DNA sequence
            special_type: DNASeq
        targets:
            shape: (None,)
            doc: (optional) values following the bed-entries
    """

    def __init__(self,
                 tsv_file,
                 trim_adapters=False,
                 sequence_length=110,
                 label_dtype=None,
                 label_start_column=1,
                 label_stop_column=None,
                 fit_sequence=False,
                 alphabet_axis=1,
                 dummy_axis=None,
                 alphabet="ACGT",
                 ignore_targets=False,
                 dtype=None):
        # core dataset, not using the one-hot encoding params
        self.seq_dl = StringDreamChallange1D(tsv_file, 
                    trim_adapters=trim_adapters, sequence_length =sequence_length, 
                    label_dtype=label_dtype, label_start_column = label_start_column, label_stop_column = label_stop_column, fit_sequence = fit_sequence, ignore_targets=ignore_targets)

        self.input_transform = ReorderedOneHot(alphabet=alphabet,
                                               dtype=dtype,
                                               alphabet_axis=alphabet_axis,
                                               dummy_axis=dummy_axis)

    def __len__(self):
        return len(self.seq_dl)

    def __getitem__(self, idx):
        ret = self.seq_dl[idx]
        ret['inputs'] = self.input_transform(str(ret["inputs"]))
        return ret

    @classmethod
    def get_output_schema(cls):
        """Get the output schema. Overrides the default `cls.output_schema`
        """
        output_schema = deepcopy(cls.output_schema)

        # get the default kwargs
        kwargs = default_kwargs(cls)

        # figure out the input shape
        mock_input_transform = ReorderedOneHot(alphabet=kwargs['alphabet'],
                                               dtype=kwargs['dtype'],
                                               alphabet_axis=kwargs['alphabet_axis'],
                                               dummy_axis=kwargs['dummy_axis'])
        input_shape = mock_input_transform.get_output_shape(
            kwargs['auto_resize_len'])

        # modify it
        output_schema.inputs.shape = input_shape

        # (optionally) get rid of the target shape
        if kwargs['ignore_targets']:
            output_schema.targets = None

        return output_schema
