import torch
from torch import nn

import config


class OCRLoss(nn.Module):
    def __init__(self, coefficients):
        super(OCRLoss, self).__init__()
        self.coefficients = coefficients

        nc = config.N_KINDS_OF_CHARACTERS
        self.resp_indexes = [5*i for i in range(nc)]
        self.coor_indexes = [5*i+1 for i in range(nc)] + [5*i+2 for i in range(nc)]
        self.size_indexes = [5*i+3 for i in range(nc)] + [5*i+4 for i in range(nc)]

    def forward(self, output, label):
        output_resp, output_coor, output_size = self.slice_tensor(output)
        label_resp, label_coor, label_size = self.slice_tensor(label)

        mask = label_resp.clone()
        ratio_of_one = mask.sum() / mask.numel()
        weight_for_resp = mask.clone()
        weight_for_resp[weight_for_resp < 0.5] = ratio_of_one * 0.5
        mask_for_bbox = torch.cat([mask.clone(), mask.clone()], dim=1)

        loss_resp = weight_for_resp * (output_resp-label_resp).pow_(2)
        loss_coor = mask_for_bbox * (output_coor-label_coor).pow_(2)
        loss_size = mask_for_bbox * (output_size.sqrt()-label_size.sqrt()).pow_(2)

        loss_resp = loss_resp.sum(dim=[1, 2, 3]).mean()
        loss_coor = loss_coor.sum(dim=[1, 2, 3]).mean()
        loss_size = loss_size.sum(dim=[1, 2, 3]).mean()

        lambda_resp, lambda_coor, lambda_size = self.coefficients

        loss = (lambda_resp*loss_resp
                + lambda_coor*loss_coor
                + lambda_size*loss_size)
        return loss, [loss_resp.item(), loss_coor.item(), loss_size.item()]

    def slice_tensor(self, tensor):
        epsilon = 1e-6
        tensor_resp = tensor[:, self.resp_indexes]
        tensor_coor = tensor[:, self.coor_indexes]
        tensor_size = tensor[:, self.size_indexes] + epsilon
        return tensor_resp, tensor_coor, tensor_size
