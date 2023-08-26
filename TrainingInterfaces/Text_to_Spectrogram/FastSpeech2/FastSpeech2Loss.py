"""
Taken from ESPNet
"""

import torch

from Layers.DurationPredictor import DurationPredictorLoss
from Utility.utils import make_non_pad_mask


def weights_nonzero_speech(target):
    # target : B x T x mel
    # Assign weight 1.0 to all labels except for padding (id=0).
    dim = target.size(-1)
    return target.abs().sum(-1, keepdim=True).ne(0).float().repeat(1, 1, dim)


class FastSpeech2Loss(torch.nn.Module):
    def __init__(
        self, use_masking=True, use_weighted_masking=False, sample_wise_loss=False
    ):
        """
        use_masking (bool):
            Whether to apply masking for padded part in loss calculation.
        use_weighted_masking (bool):
            Whether to weighted masking in loss calculation.
        """
        super().__init__()

        assert (use_masking != use_weighted_masking) or not use_masking
        self.use_masking = use_masking
        self.use_weighted_masking = use_weighted_masking
        self.sample_wise_loss = (
            True  # ALWAYS True IN ORDER TO TRACK SAMPLE BASED ON DATASET_ID
        )

        # define criterions
        reduction = (
            "none" if self.use_weighted_masking or self.sample_wise_loss else "mean"
        )

        self.l1_criterion = torch.nn.L1Loss(reduction=reduction)
        self.mse_criterion = torch.nn.MSELoss(reduction=reduction)
        self.duration_criterion = DurationPredictorLoss(reduction=reduction)

    def forward(
        self,
        after_outs,
        before_outs,
        d_outs,
        p_outs,
        e_outs,
        ys,
        ds,
        ps,
        es,
        ilens,
        olens,
        dataset_ids=None,
    ):
        """
        Args:
            after_outs (Tensor): Batch of outputs after postnets (B, Lmax, odim).
            before_outs (Tensor): Batch of outputs before postnets (B, Lmax, odim).
            d_outs (LongTensor): Batch of outputs of duration predictor (B, Tmax).
            p_outs (Tensor): Batch of outputs of pitch predictor (B, Tmax, 1).
            e_outs (Tensor): Batch of outputs of energy predictor (B, Tmax, 1).
            ys (Tensor): Batch of target features (B, Lmax, odim).
            ds (LongTensor): Batch of target durations (B, Tmax).
            ps (Tensor): Batch of target token-averaged pitch (B, Tmax, 1).
            es (Tensor): Batch of target token-averaged energy (B, Tmax, 1).
            ilens (LongTensor): Batch of the lengths of each input (B,).
            olens (LongTensor): Batch of the lengths of each target (B,).
            dataset_ids (LongTensor): Batch of dataset_ids (cf. Utility/map_dataset_id.py) (B,).

        Returns:
            Tensor: L1 loss value.
            Tensor: Duration predictor loss value.
            Tensor: Pitch predictor loss value.
            Tensor: Energy predictor loss value.
            List: contains the lists (with sample-wise elements) dataset_ids, l1_losses, duration_losses, pitch_losses, energy_losses.

        """
        # apply masks to remove padded parts
        if self.use_masking:
            out_masks = make_non_pad_mask(olens).unsqueeze(-1).to(ys.device)
            before_outs = before_outs.masked_select(out_masks)
            if after_outs is not None:
                after_outs = after_outs.masked_select(out_masks)
            ys = ys.masked_select(out_masks)
            duration_masks = make_non_pad_mask(ilens).to(ys.device)
            d_outs = d_outs.masked_select(duration_masks)
            ds = ds.masked_select(duration_masks)
            pitch_masks = make_non_pad_mask(ilens).unsqueeze(-1).to(ys.device)
            p_outs = p_outs.masked_select(pitch_masks)
            e_outs = e_outs.masked_select(pitch_masks)
            ps = ps.masked_select(pitch_masks)
            es = es.masked_select(pitch_masks)

        # calculate loss
        l1_loss = self.l1_criterion(before_outs, ys)
        if after_outs is not None:
            l1_loss = l1_loss + self.l1_criterion(after_outs, ys)
        duration_loss = self.duration_criterion(d_outs, ds)
        pitch_loss = self.mse_criterion(p_outs, ps)
        energy_loss = self.mse_criterion(e_outs, es)

        # make weighted mask and apply it
        if self.use_weighted_masking:
            out_masks = make_non_pad_mask(olens).unsqueeze(-1).to(ys.device)
            out_masks = torch.nn.functional.pad(
                out_masks.transpose(1, 2),
                [0, ys.size(1) - out_masks.size(1), 0, 0, 0, 0],
                value=False,
            ).transpose(1, 2)

            out_weights = out_masks.float() / out_masks.sum(dim=1, keepdim=True).float()
            out_weights /= ys.size(0) * ys.size(2)
            duration_masks = make_non_pad_mask(ilens).to(ys.device)
            duration_weights = (
                duration_masks.float() / duration_masks.sum(dim=1, keepdim=True).float()
            )
            duration_weights /= ds.size(0)

            # apply weight
            l1_loss = l1_loss.mul(out_weights).masked_select(out_masks).sum()
            duration_loss = (
                duration_loss.mul(duration_weights).masked_select(duration_masks).sum()
            )
            pitch_masks = duration_masks.unsqueeze(-1)
            pitch_weights = duration_weights.unsqueeze(-1)
            pitch_loss = pitch_loss.mul(pitch_weights).masked_select(pitch_masks).sum()
            energy_loss = (
                energy_loss.mul(pitch_weights).masked_select(pitch_masks).sum()
            )

        if self.sample_wise_loss:
            sample_wise_l1_loss = list()
            sample_wise_duration_loss = list()
            sample_wise_pitch_loss = list()
            sample_wise_energy_loss = list()

            if after_outs:
                sample_wise_l1_loss_after_outs = list()
                for i in range(ys.size(0)):
                    sample_y = after_outs[i]
                    sample_y_target = ys[i]
                    sample_wise_l1_loss_after_outs.append(
                        self.l1_criterion(sample_y, sample_y_target)
                    )

            assert (
                ys.size(0) == (ds.size(0)) == ps.size(0) == es.size(0)
            ), "Error: Mismatching number of y, pitch, duration, energy values."

            # apply sample-wise weighted masking (because every sample is still padded to the max length sample in the batch)
            for i in range(ys.size(0)):
                sample_y = before_outs[i]
                sample_y_target = ys[i]
                sample_wise_l1_loss.append(self.l1_criterion(sample_y, sample_y_target))
                if after_outs:
                    sample_wise_l1_loss[i] += sample_wise_l1_loss_after_outs[i]
                sample_wise_l1_loss[i] = (
                    sample_wise_l1_loss[i]
                    .mul(out_weights[i])
                    .masked_select(out_masks[i])
                    .sum()
                )

                sample_d = d_outs[i]
                sample_d_target = ds[i]
                sample_wise_duration_loss.append(
                    self.duration_criterion(sample_d, sample_d_target)
                )
                sample_wise_duration_loss[i] = (
                    sample_wise_duration_loss[i]
                    .mul(duration_weights[i])
                    .masked_select(duration_masks[i])
                    .sum()
                )

                sample_p = p_outs[i]
                sample_p_target = ps[i]
                sample_wise_pitch_loss.append(
                    self.mse_criterion(sample_p, sample_p_target)
                )
                sample_wise_pitch_loss[i] = (
                    sample_wise_pitch_loss[i]
                    .mul(pitch_weights[i])
                    .masked_select(pitch_masks[i])
                    .sum()
                )

                sample_e = e_outs[i]
                sample_e_target = es[i]
                sample_wise_energy_loss.append(
                    self.mse_criterion(sample_e, sample_e_target)
                )
                sample_wise_energy_loss[i] = (
                    sample_wise_energy_loss[i]
                    .mul(pitch_weights[i])
                    .masked_select(pitch_masks[i])
                    .sum()
                )

            sample_wise_losses_with_dataset_ids = [dataset_ids]
            sample_wise_losses_with_dataset_ids.append(sample_wise_l1_loss)
            sample_wise_losses_with_dataset_ids.append(sample_wise_duration_loss)
            sample_wise_losses_with_dataset_ids.append(sample_wise_pitch_loss)
            sample_wise_losses_with_dataset_ids.append(sample_wise_energy_loss)

        return (
            l1_loss,
            duration_loss,
            pitch_loss,
            energy_loss,
            sample_wise_losses_with_dataset_ids,
        )
