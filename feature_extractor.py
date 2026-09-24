# feature extractors: nnuNet

# imports:
import torch
from torch import nn
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor


################# nnUNet ########################
def get_pretrained_nnunet(model_path="path_to_pretrained_nnunet", frozen_decoder = True):
    """
    Load a pretrained nnU-Net model and extract only the encoder-decoder weights.
    """
    predictor = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
    )
    # Load pretrained nnU-Net model
    predictor.initialize_from_trained_model_folder(
        model_path,  # Path to the pretrained model
        use_folds=(0,)
    )
    model = predictor.network  # Get the full model

    for head_name in ("seg_layers", "final_conv", "segmentation_head"):
        if hasattr(model, head_name):
            setattr(model, head_name, nn.Identity())
            break

    class EncoderDecoder(nn.Module):
        def __init__(self, net):
            super().__init__()
            # in nnUNet v2, the top‐level network has exactly these two submodules:
            self.encoder = net.encoder
            self.decoder = net.decoder

            for param in self.encoder.parameters():
                param.requires_grad = False

            if frozen_decoder:
                for param in self.decoder.parameters():
                    param.requires_grad = False

                # unfreeze decoder
                for p in self.decoder.stages[-1].parameters():
                    p.requires_grad_(True)

            else:
                for param in self.decoder.stages.parameters():
                    param.requires_grad = True

        def forward(self, x):
            # `encoder` returns a list of feature maps at each stage
            feats = self.encoder(x)
            # `decoder` knows how to take that list and do its skip‐connected upsampling
            out = self.decoder(feats)
            return out
    #

    return EncoderDecoder(model)

def hook_(module, inp, out):
    # inp is a tuple of inputs; for Conv3d it's always (x,)
    module.post = out # DO NOT HAVE DETACH!


class nnUnet_feature_maps_multiple(nn.Module):
    '''
    purpose: to extract the feature maps right before the 1x1x1 convolution in nnUNet. the second to last section of the
    decoder should give 32 feature maps in the size of the original output
    '''

    def __init__(self,model_path="path_to_pretrained_nnunet", frozen_decoder = True):
        super().__init__()
        self.encoder_decoder = get_pretrained_nnunet(model_path, frozen_decoder)

    def forward(self, image):

        hook_handle1 = self.encoder_decoder.decoder.stages[-1].register_forward_hook(hook_)
        hook_handle2 = self.encoder_decoder.decoder.stages[-2].register_forward_hook(hook_)
        hook_handle3 = self.encoder_decoder.decoder.stages[-3].register_forward_hook(hook_)
        hook_handle4 = self.encoder_decoder.decoder.stages[-4].register_forward_hook(hook_)
        hook_handle5 = self.encoder_decoder.decoder.stages[-5].register_forward_hook(hook_)
        hook_handle6 = self.encoder_decoder.decoder.stages[-6].register_forward_hook(hook_)

        _ = self.encoder_decoder(image)
        feature_maps1 = self.encoder_decoder.decoder.stages[-1].post
        feature_maps2 = self.encoder_decoder.decoder.stages[-2].post
        feature_maps3 = self.encoder_decoder.decoder.stages[-3].post
        feature_maps4 = self.encoder_decoder.decoder.stages[-4].post
        feature_maps5 = self.encoder_decoder.decoder.stages[-5].post
        feature_maps6 = self.encoder_decoder.decoder.stages[-6].post

        hook_handle1.remove()
        hook_handle2.remove()
        hook_handle3.remove()
        hook_handle4.remove()
        hook_handle5.remove()
        hook_handle6.remove()

        return feature_maps1, feature_maps2, feature_maps3, feature_maps4, feature_maps5, feature_maps6



