# -*- coding: utf-8 -*-

"""
The SAMRI model
Reference: The model is referenced from the segment anything model, 
            link: https://github.com/facebookresearch/segment-anything
"""

import torch
from torch.nn import functional as F

from typing import Any, Dict, List

from segment_anything.modeling.image_encoder import ImageEncoderViT
from segment_anything.modeling.mask_decoder import MaskDecoder
from segment_anything.modeling.prompt_encoder import PromptEncoder

from segment_anything.modeling import Sam
# from skimage import exposure
from torchvision.transforms.functional import equalize

class SAMRI(Sam):
    def __init__(
        self,
        image_encoder: ImageEncoderViT,
        mask_decoder: MaskDecoder,
        prompt_encoder: PromptEncoder,
    ) -> None:
        super().__init__(image_encoder = image_encoder,
                         prompt_encoder = prompt_encoder,
                         mask_decoder = mask_decoder
                         )
        # freeze prompt encoder
        for param in self.prompt_encoder.parameters():
            param.requires_grad = False
        # freeze image encoder
        for param in self.image_encoder.parameters():
            param.requires_grad = False


    def forward(
            self,
            batched_input: List[Dict[str, Any]],
            multimask_output: bool,
            train_mode: bool = False
        ) -> List[Dict[str, torch.Tensor]]:
        """
        Predicts masks end-to-end from provided images and prompts.
        If prompts are not known in advance, using SamPredictor is
        recommended over calling the model directly.

        Arguments:
          batched_input (list(dict)): A list over input images, each a
            dictionary with the following keys. A prompt key can be
            excluded if it is not present.
              'image': The image as a torch tensor in 3xHxW format,
                already transformed for input to the model.
              'original_size': (tuple(int, int)) The original size of
                the image before transformation, as (H, W).
              'point_coords': (torch.Tensor) Batched point prompts for
                this image, with shape BxNx2. Already transformed to the
                input frame of the model.
              'point_labels': (torch.Tensor) Batched labels for point prompts,
                with shape BxN.
              'boxes': (torch.Tensor) Batched box inputs, with shape Bx4.
                Already transformed to the input frame of the model.
              'mask_inputs': (torch.Tensor) Batched mask inputs to the model,
                in the form Bx1xHxW.
          multimask_output (bool): Whether the model should predict multiple
            disambiguating masks, or return a single mask.

        Returns:
          (list(dict)): A list over input images, where each element is
            as dictionary with the following keys.
              'masks': (torch.Tensor) Batched binary mask predictions,
                with shape BxCxHxW, where B is the number of input prompts,
                C is determined by multimask_output, and (H, W) is the
                original size of the image.
              'iou_predictions': (torch.Tensor) The model's predictions
                of mask quality, in shape BxC.
              'low_res_logits': (torch.Tensor) Low resolution logits with
                shape BxCxHxW, where H=W=256. Can be passed as mask input
                to subsequent iterations of prediction.
        """
        
        input_images = torch.stack([self.preprocess(x["image"]) for x in batched_input], dim=0)
        image_embeddings = self.image_encoder(input_images)

        if "point_coords" in batched_input:
            if batched_input["point_coords"] != None:
                point_coords = [point["point_coords"] for point in batched_input]
                point_labels = [label["point_labels"] for label in batched_input]
                points = (torch.stack(point_coords), torch.stack(point_labels))
            else:
                points = None
        else:
            points = None

        if "bbox" in batched_input:
            if batched_input["bbox"] != None:
                bboxes = [box["boxes"][None, :] for box in batched_input]
                bboxes = torch.stack(bboxes, dim=0)
            else:
                bboxes = None
        else:
            bboxes = None
        
        sparse_embeddings, dense_embeddings = self.prompt_encoder(
            points=points,
            boxes=bboxes,
            masks=None,
        )
        low_res_masks, iou_predictions = self.mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=self.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=multimask_output,
        )
        masks = self.postprocess_masks(
            low_res_masks,
            input_size=input_images.shape[-2:],
            original_size=batched_input[0]["original_size"],
        )
        
        if train_mode:
            return masks
        else:
            masks = masks > self.mask_threshold
            return masks

    