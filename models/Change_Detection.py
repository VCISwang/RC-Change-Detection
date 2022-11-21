import math, time
from itertools import chain
import torch
import torch.nn.functional as F
from torch import nn
from models.base_model import BaseModel
from models.decoders import *
from models.encoder import Encoder
from copy import deepcopy
from torch.distributions.uniform import Uniform


class CD_Model(BaseModel):
    def __init__(self, num_classes, pretrained=True):

        self.num_classes = num_classes
        super(CD_Model, self).__init__()

        # Create the model
        self.encoder = Encoder(pretrained=pretrained)

        # The main encoder
        upscale = 8
        num_out_ch = 2048
        decoder_in_ch = num_out_ch // 4
        self.main_decoder = MainDecoder(upscale, decoder_in_ch, num_classes=num_classes)

        self.aux = False
        # The auxilary decoders
        # if self.mode == 'semi' or self.mode == 'weakly_semi':
        #     vat_decoder = [VATDecoder(upscale, decoder_in_ch, num_classes, xi=conf['xi'],
        #                               eps=conf['eps']) for _ in range(conf['vat'])]
            # drop_decoder = [DropOutDecoder(upscale, decoder_in_ch, num_classes,
            #                                drop_rate=conf['drop_rate'], spatial_dropout=conf['spatial'])
            #                 for _ in range(conf['drop'])]
            # cut_decoder = [CutOutDecoder(upscale, decoder_in_ch, num_classes, erase=conf['erase'])
            #                for _ in range(conf['cutout'])]
            # context_m_decoder = [ContextMaskingDecoder(upscale, decoder_in_ch, num_classes)
            #                      for _ in range(conf['context_masking'])]
            # object_masking = [ObjectMaskingDecoder(upscale, decoder_in_ch, num_classes)
            #                   for _ in range(conf['object_masking'])]
            # feature_drop = [FeatureDropDecoder(upscale, decoder_in_ch, num_classes)
            #                 for _ in range(conf['feature_drop'])]
            # feature_noise = [FeatureNoiseDecoder(upscale, decoder_in_ch, num_classes,
            #                                      uniform_range=conf['uniform_range'])
            #                  for _ in range(conf['feature_noise'])]
            #
            # self.aux_decoders = nn.ModuleList([*vat_decoder, *drop_decoder, *cut_decoder,
            #                                    *context_m_decoder, *object_masking, *feature_drop, *feature_noise])
            # self.aux_decoders = nn.ModuleList([*vat_decoder])

    def forward(self, A_l=None, B_l=None):

        output_l = self.main_decoder(self.encoder(A_l, B_l))

        return output_l

    def get_backbone_params(self):
        return self.encoder.get_backbone_params()

    def get_other_params(self):
        if self.aux:
            return chain(self.encoder.get_module_params(), self.main_decoder.parameters())
            # return chain(self.encoder.get_module_params(), self.main_decoder.parameters(),
            #              self.aux_decoders.parameters())

        return chain(self.encoder.get_module_params(), self.main_decoder.parameters())

