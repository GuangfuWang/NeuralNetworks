# Author: Guangfu Wang
# Date: 2022-01-24
# CopyRight Guangfu

import numpy as np
from CommonBlock.NumpyVersion.ActivationBlock.ActivationFactory import ActivationFactory
from Transformer.NumpyVersion.MultiHeadSelfAttentionBlock import NumpyMultiHeadSelfAttention as MSA
from CommonBlock.NumpyVersion.Norm import LayerNorm as LN
from CommonBlock.NumpyVersion.Network import FullyConnectBlock as FCB


class NumpyVanillaEncoder:
    def __init__(self):
        self.msa = MSA()
        self.norm1 = LN()
        self.fc = FCB()
        self.norm2 = LN()
