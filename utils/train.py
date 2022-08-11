import os
from tqdm import tqdm

import torch
import torch.nn.functional as F

from metrics.DiceScore import mDSC
from utils.one_hot_encoder import one_hot_encode, reverse_one_hot

import matplotlib.pyplot as plt
import seaborn as sns

sns.set(font_scale=1.2, style='whitegrid')


def save_states(segment_model, optimizer, history, epoch, state_dict_path):
    torch.save({
        "model": segment_model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "history": history,
        "epoch": epoch,
    }, state_dict_path)


def plot_train_history(history, batch, segment_model, device):
    # Отрисовка графиков
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history["train_loss"], label='train')
    plt.plot(history["val_loss"], label='test')
    plt.title("Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history["train_mDSC"], label='train')
    plt.plot(history["val_mDSC"], label='test')
    plt.title("mDSC")
    plt.legend()

    plt.show()

    # Отрисовка результатов
    n_img = 4

    plt.figure(figsize=(20, n_img * 3.5))

    for i, (image, mask) in batch:
        if i == n_img:
            break

        plt.subplot(n_img, 3, 3 * i + 1)
        plt.imshow(image.permute(1, 2, 0).cpu())
        plt.axis("off")
        plt.title("Изображение")

        plt.subplot(n_img, 3, 3 * i + 2)
        plt.imshow(reverse_one_hot(mask.squeeze().cpu().permute((1, 2, 0))))
        plt.axis("off")
        plt.title("Реальная маска")

        with torch.no_grad():
            pred = segment_model(image.unsqueeze(dim=0).to(device)) \
                .squeeze().cpu()

            pred_labels = torch.argmax(F.softmax(pred.permute((1, 2, 0)), dim=-1), dim=-1)

        plt.subplot(n_img, 3, 3 * i + 3)
        plt.imshow(pred_labels)
        plt.axis("off")
        plt.title("Предсказанная маска")


def train(
        segment_model,
        optimizer,
        criterion,
        train_dataloader,
        val_dataloader,
        state_dict_path,
        device="cpu",
        n_epochs=20,
        show_interval=20,
        savefig_dir=None,
):
    history = {}
    history["train_loss"] = []
    history["train_mDSC"] = []
    history["val_loss"] = []
    history["val_mDSC"] = []

    start_epoch = 0

    # Загрузим последнее состояние
    if os.path.exists(state_dict_path):
        state = torch.load(state_dict_path)
        segment_model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        history = state["history"]
        start_epoch = state["epoch"] + 1

    n_train_batches = len(train_dataloader)
    n_val_batches = len(val_dataloader)

    best_val_mDSC = 0.

    end_epoch = start_epoch + n_epochs

    if savefig_dir is not None:
        if not os.path.exists(savefig_dir):
            os.mkdir(savefig_dir)

    for epoch in range(start_epoch, start_epoch + n_epochs):

        print(f"Epoch {epoch}/{end_epoch}")

        segment_model.train()

        train_loss = 0.
        train_mDSC = 0.

        for i, (image, mask) in enumerate(tqdm(train_dataloader)):
            image = image.to(device)
            mask = mask.to(device)

            pred = segment_model(image)
            loss = criterion(pred, mask.float())
            loss.backward()

            if i == (n_train_batches - 1):
                optimizer.step()
                optimizer.zero_grad()

            loss_ = float(loss.detach().data)

            pred_labels = torch.argmax(F.softmax(pred.detach().permute((0, 2, 3, 1)), dim=-1), dim=-1)
            pred_labels = one_hot_encode(pred_labels)
            mDSC_ = float(mDSC(pred_labels, mask.permute((0, 2, 3, 1))).data)

            train_loss += loss_
            train_mDSC += mDSC_

        train_loss = train_loss / n_train_batches
        train_mDSC = train_mDSC / n_train_batches

        history["train_loss"].append(train_loss)
        history["train_mDSC"].append(train_mDSC)

        print('')
        print(f"Total Train:\tloss\t{train_loss:.5f}"
              f"\t\tmDSC\t{train_mDSC:.5f}")

        segment_model.eval()

        val_loss = 0.
        val_mDSC = 0.

        with torch.no_grad():
            for image, mask in tqdm(val_dataloader):
                image, mask = image.to(device), mask.to(device)
                pred = segment_model(image)
                loss = criterion(pred, mask.float())
                loss_ = float(loss.data)

                pred_labels = torch.argmax(F.softmax(pred.permute((0, 2, 3, 1)), dim=-1), dim=-1)
                pred_labels = one_hot_encode(pred_labels)
                mDSC_ = float(mDSC(pred_labels, mask.permute((0, 2, 3, 1))).data)

                val_mDSC += mDSC_
                val_loss += loss_

        val_loss = val_loss / n_val_batches
        val_mDSC = val_mDSC / n_val_batches

        history["val_loss"].append(val_loss)
        history["val_mDSC"].append(val_mDSC)

        if history["val_mDSC"][-1] > best_val_mDSC:
            save_states(segment_model, optimizer, history, epoch, state_dict_path)

            torch.save({
                "model": segment_model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "history": history,
                "epoch": epoch,
            }, state_dict_path)
            best_val_mDSC = history["val_mDSC"][-1]

        print('')
        print(f"Total Valid:\tloss\t{val_loss:.5f}"
              f"\t\tmDSC\t{val_mDSC:.5f}"
              f"\t\tbest mDSC\t{best_val_mDSC:.5f}")
        print('-' * 100)

        if epoch % show_interval == show_interval - 1:

            batch = next(iter(val_dataloader))
            plot_train_history(history, batch, segment_model, device)

            if savefig_dir is not None:
                plt.savefig(os.path.join(savefig_dir, f"results_epoch_{epoch:03d}.png"))
            plt.show()
