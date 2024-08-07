import argparse
import os
import csv
import cv2
import copy
import tqdm
import torch
import warnings
import numpy as np
from PIL import Image
from timm import utils
from torchvision import transforms
from torch.utils.data import DataLoader

from utils import util
from nets.nn import U2NETP
from utils.dataset import Dataset

warnings.filterwarnings("ignore")


def train(args):
    util.setup_seed()
    model = U2NETP().cuda()

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)

    # Scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-5)

    dataset = Dataset(os.path.join(args.data_dir, 'train'), transform=transforms.Compose([
        util.RescaleT(320),
        util.RandomCrop(288),
        util.ToTensorLab(flag=0)]))
    sampler = None
    if args.distributed:
        sampler = torch.utils.data.distributed.DistributedSampler(dataset)

    loader = DataLoader(dataset, args.batch_size, not args.distributed,
                        sampler, num_workers=8, pin_memory=True, drop_last=True)

    if args.distributed:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = torch.nn.parallel.DistributedDataParallel(module=model,
                                                          device_ids=[args.local_rank])

    best = float('inf')
    num_batch = len(loader)
    amp_scale = torch.cuda.amp.GradScaler()
    with open('weights/logs.csv', 'w') as f:
        if args.local_rank == 0:
            writer = csv.DictWriter(f, fieldnames=['epoch', 'Loss'])
            writer.writeheader()
        for epoch in range(args.epochs):
            model.train()
            p_bar = enumerate(loader)
            avg_loss1 = util.AverageMeter()
            avg_loss2 = util.AverageMeter()

            if args.local_rank == 0:
                print(('\n' + '%10s' * 2) % ('epoch', 'loss'))
                p_bar = tqdm.tqdm(iterable=p_bar, total=num_batch)

            for i, data in p_bar:
                images, labels = data['image'], data['label']

                images = images.cuda().float()
                labels = labels.cuda().float()

                optimizer.zero_grad()

                output = model(images)
                loss, losses = util.loss_fusion(output, labels)
                # Backward
                amp_scale.scale(losses).backward()

                # Optimize
                amp_scale.unscale_(optimizer)
                amp_scale.step(optimizer)
                amp_scale.update()
                optimizer.zero_grad()

                # Log
                if args.distributed:
                    loss = utils.reduce_tensor(losses.data, args.world_size)

                avg_loss1.update(losses.data.item(), images.size(0))
                avg_loss2.update(loss.data.item(), images.size(0))

                if args.local_rank == 0:
                    s = ('%10s' + '%10.4g') % (f'{epoch + 1}/{args.epochs}', losses.item())
                    p_bar.set_description(s)

            scheduler.step()
            if args.local_rank == 0:
                last = test(args, copy.deepcopy(model.module if args.distributed else model))
                writer.writerow({'Loss': str(f'{last:.3f}'),
                                 'epoch': str(epoch + 1).zfill(3)})

                f.flush()

                if best > last:
                    best = last

                # Model Save
                ckpt = {'model': copy.deepcopy(model.module if args.distributed else model).half()}

                # Save last and best result
                torch.save(ckpt, './weights/last.pt')
                if best == last:
                    torch.save(ckpt, './weights/best.pt')
                del ckpt
                print(f"Best Loss = {best:.3f}")

            del output, loss, losses

    if args.local_rank == 0:
        util.strip_optimizer('./weights/best.pt')
        util.strip_optimizer('./weights/last.pt')

    torch.cuda.empty_cache()


def test(args, model=None):
    if model is None:
        model = torch.load('weights/best.pt', map_location='cuda')['model'].float()
    model.half()
    model.cuda()
    model.eval()

    dataset = Dataset(os.path.join(args.data_dir, 'test'), transform=transforms.Compose([
        util.RescaleT(320),
        util.ToTensorLab(flag=0)]))

    avg_loss = util.AverageMeter()
    loader = DataLoader(dataset, args.batch_size, num_workers=4)
    with torch.no_grad():
        for data in tqdm.tqdm(loader, '%20s' % 'Loss'):
            images, labels = data['image'], data['label']
            images = images.cuda().half()
            labels = labels.cuda().half()

            output = model(images)
            _, loss = util.loss_fusion(output, labels)

            avg_loss.update(loss.data.item(), images.size(0))

    print(f"Last Loss = {avg_loss.avg:.3f}")
    model.float()  # for training
    return avg_loss.avg


def demo():
    model = torch.load('weights/best.pt', map_location='cuda')['model'].float()
    model.cuda()
    model.eval()

    stream = cv2.VideoCapture('images/video.mp4')
    fps = int(stream.get(cv2.CAP_PROP_FPS))
    h = int(stream.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w = int(stream.get(cv2.CAP_PROP_FRAME_WIDTH))
    fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')

    output = cv2.VideoWriter('demo/demo1.avi', fourcc, fps, (w, h))

    if not stream.isOpened():
        print("Error: Could not open webcam.")
        return

    transform = transforms.Compose([util.RescaleT(320), util.ToTensorLab(flag=0)])

    while True:
        ret, frame = stream.read()
        if not ret:
            break
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        sample = {'image': np.array(image), 'label': np.zeros_like(np.array(image))}

        image = transform(sample)['image'].unsqueeze(0)
        image = image.type(torch.FloatTensor).cuda()

        with torch.inference_mode():
            d1, d2, d3, d4, d5, d6, d7 = model(image)

        pred = d1[:, 0, :, :]
        pred = util.normPRED(pred).squeeze().cpu().data.numpy()

        pred_image = (pred * 255).astype(np.uint8)
        pred_image = cv2.cvtColor(pred_image, cv2.COLOR_GRAY2BGR)
        pred_image = cv2.resize(pred_image, (frame.shape[1], frame.shape[0]))
        cv2.imshow('pred image', pred_image)
        output.write(pred_image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    stream.release()
    output.release()
    cv2.destroyAllWindows()


def demo_image():
    model = torch.load('weights/last.pt', map_location='cuda')['model'].float()
    model.cuda()
    model.eval()

    transform = transforms.Compose([util.RescaleT(320), util.ToTensorLab(flag=0)])

    frame = cv2.imread('images/5.png')
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    sample = {'image': np.array(image), 'label': np.zeros_like(np.array(image))}

    image = transform(sample)['image'].unsqueeze(0)
    image = image.type(torch.FloatTensor).cuda()

    with torch.inference_mode():
        d1, d2, d3, d4, d5, d6, d7 = model(image)

    pred = d1[:, 0, :, :]
    pred = util.normPRED(pred).squeeze().cpu().data.numpy()

    pred_image = (pred * 255).astype(np.uint8)
    pred_image = cv2.cvtColor(pred_image, cv2.COLOR_GRAY2BGR)
    pred_image = cv2.resize(pred_image, (frame.shape[1], frame.shape[0]))
    cv2.imshow('pred image', pred_image)
    cv2.imwrite('demo/5.png', pred_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='../../Datasets/SOD')
    parser.add_argument('--epochs', type=int, default=10000)
    parser.add_argument('--local-rank', default=0, type=int)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--demo', default=True, action='store_true')
    parser.add_argument('--demo-image', action='store_true')
    args = parser.parse_args()

    args.world_size = int(os.getenv('WORLD_SIZE', 1))
    args.distributed = int(os.getenv('WORLD_SIZE', 1)) > 1

    if args.distributed:
        torch.cuda.set_device(device=args.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')

    if args.train:
        train(args)
    if args.test:
        test(args)
    if args.demo:
        demo()
    if args.demo_image:
        demo_image()


if __name__ == "__main__":
    main()
