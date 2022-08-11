import torch


def DSC(predicted_mask_channel, target_mask_channel):
    """ Считает Dice Score для одного канала """
    # Площадь пересечения в пикселях
    intersection = (predicted_mask_channel & target_mask_channel).sum(dim=(1, 2))
    # Сумма площадей в пикселях
    sum_ = predicted_mask_channel.sum(dim=(1, 2)) + target_mask_channel.sum(dim=(1, 2))
    dsc = (2 * intersection / sum_).mean()
    return dsc


def mDSC(predicted_mask_batch, target_mask_batch):
    """
    Считает Dice Score для всех классов батча
    : input_shape = (batch_size, width, height, n_channels)
    """

    n_channels = predicted_mask_batch.shape[1]
    n_classes = 0
    mDSC_ = 0.

    for channel in range(1, n_channels):
        predicted_mask_channel = predicted_mask_batch[:, :, :, channel]
        target_mask_channel = target_mask_batch[:, :, :, channel]

        dsc = DSC(predicted_mask_channel, target_mask_channel)
        if not torch.isnan(dsc).item():
            n_classes += 1

        mDSC_ += torch.nan_to_num(dsc)

    mDSC_ /= n_classes

    return mDSC_
