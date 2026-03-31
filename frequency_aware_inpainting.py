import math
import os
import random
from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.utils import save_image


# ============================================================
# 1. Config
# ============================================================


@dataclass
class Config:
    data_root: str = "./data"  # e.g. Places2 root or any image folder root
    image_size: int = 256
    batch_size: int = 8
    num_workers: int = 4
    lr: float = 2e-4
    weight_decay: float = 1e-4
    epochs: int = 20
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    save_dir: str = "./outputs"
    log_interval: int = 50

    # loss weights
    lambda_rec: float = 1.0
    lambda_mask: float = 6.0
    lambda_map: float = 1.0
    lambda_wavelet: float = 0.5
    lambda_sparse: float = 1e-3

    # model width
    base_ch: int = 64

    # threshold for visualization / possible future routing
    importance_threshold: float = 0.5


CFG = Config()
os.makedirs(CFG.save_dir, exist_ok=True)


# ============================================================
# 2. Random Mask Generator
# ============================================================


class RandomIrregularMask:
    def __init__(self, image_size: int, min_holes: int = 4, max_holes: int = 10):
        self.image_size = image_size
        self.min_holes = min_holes
        self.max_holes = max_holes

    def __call__(self) -> torch.Tensor:
        h = w = self.image_size
        mask = torch.zeros(1, h, w, dtype=torch.float32)

        num_holes = random.randint(self.min_holes, self.max_holes)
        for _ in range(num_holes):
            hole_type = random.choice(["rect", "brush"])

            if hole_type == "rect":
                rh = random.randint(h // 8, h // 2)
                rw = random.randint(w // 8, w // 2)
                y1 = random.randint(0, h - rh)
                x1 = random.randint(0, w - rw)
                mask[:, y1 : y1 + rh, x1 : x1 + rw] = 1.0
            else:
                points = random.randint(4, 8)
                x, y = random.randint(0, w - 1), random.randint(0, h - 1)
                for _ in range(points):
                    angle = random.uniform(0, 2 * math.pi)
                    length = random.randint(h // 16, h // 6)
                    brush_w = random.randint(8, 24)
                    x2 = int(max(0, min(w - 1, x + length * math.cos(angle))))
                    y2 = int(max(0, min(h - 1, y + length * math.sin(angle))))
                    self._draw_line(mask[0], x, y, x2, y2, brush_w)
                    x, y = x2, y2

        return mask.clamp(0, 1)

    @staticmethod
    def _draw_line(mask2d: torch.Tensor, x1: int, y1: int, x2: int, y2: int, width: int):
        num_steps = max(abs(x2 - x1), abs(y2 - y1)) + 1
        xs = torch.linspace(x1, x2, steps=num_steps)
        ys = torch.linspace(y1, y2, steps=num_steps)
        radius = width // 2
        h, w = mask2d.shape
        for x, y in zip(xs, ys):
            xi, yi = int(x.item()), int(y.item())
            y_min, y_max = max(0, yi - radius), min(h, yi + radius + 1)
            x_min, x_max = max(0, xi - radius), min(w, xi + radius + 1)
            mask2d[y_min:y_max, x_min:x_max] = 1.0


# ============================================================
# 3. Dataset
# ============================================================


class InpaintingDataset(Dataset):
    def __init__(self, root: str, image_size: int):
        self.base = ImageFolder(
            root=root,
            transform=transforms.Compose(
                [
                    transforms.Resize((image_size, image_size)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
                ]
            ),
        )
        self.mask_gen = RandomIrregularMask(image_size=image_size)

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx: int):
        img, _ = self.base[idx]
        mask = self.mask_gen()  # 1: hole / 0: known
        masked_img = img * (1.0 - mask)
        return {
            "image": img,
            "mask": mask,
            "masked_image": masked_img,
        }


# ============================================================
# 4. Haar Wavelet Utilities
# ============================================================


class HaarDWT(nn.Module):
    def __init__(self):
        super().__init__()
        ll = torch.tensor([[0.5, 0.5], [0.5, 0.5]], dtype=torch.float32)
        lh = torch.tensor([[-0.5, -0.5], [0.5, 0.5]], dtype=torch.float32)
        hl = torch.tensor([[-0.5, 0.5], [-0.5, 0.5]], dtype=torch.float32)
        hh = torch.tensor([[0.5, -0.5], [-0.5, 0.5]], dtype=torch.float32)
        filt = torch.stack([ll, lh, hl, hh], dim=0).unsqueeze(1)  # [4,1,2,2]
        self.register_buffer("filt", filt)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        b, c, h, w = x.shape
        filt = self.filt.repeat(c, 1, 1, 1)  # [4C,1,2,2]
        out = F.conv2d(x, filt, stride=2, padding=0, groups=c)  # [B,4C,H/2,W/2]
        out = out.view(b, c, 4, h // 2, w // 2)
        ll = out[:, :, 0]
        lh = out[:, :, 1]
        hl = out[:, :, 2]
        hh = out[:, :, 3]
        return ll, lh, hl, hh


def build_importance_target(image: torch.Tensor, dwt: HaarDWT) -> torch.Tensor:
    """
    image: [-1,1], [B,3,H,W]
    return: [B,1,H,W] in [0,1]
    """
    _, lh, hl, hh = dwt(image)
    energy = (
        lh.abs().mean(dim=1, keepdim=True)
        + hl.abs().mean(dim=1, keepdim=True)
        + hh.abs().mean(dim=1, keepdim=True)
    )
    energy = F.interpolate(
        energy, size=image.shape[-2:], mode="bilinear", align_corners=False
    )

    # robust normalize per-sample
    e_min = energy.amin(dim=(2, 3), keepdim=True)
    e_max = energy.amax(dim=(2, 3), keepdim=True)
    norm = (energy - e_min) / (e_max - e_min + 1e-6)
    return norm.clamp(0, 1)


# ============================================================
# 5. CNN Building Blocks
# ============================================================


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UpBlock(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.conv = ConvBlock(in_ch + skip_ch, out_ch)

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class SharedEncoder(nn.Module):
    def __init__(self, in_ch: int = 4, base_ch: int = 64):
        super().__init__()
        self.e1 = ConvBlock(in_ch, base_ch)  # 256
        self.e2 = ConvBlock(base_ch, base_ch * 2, 2)  # 128
        self.e3 = ConvBlock(base_ch * 2, base_ch * 4, 2)  # 64
        self.e4 = ConvBlock(base_ch * 4, base_ch * 8, 2)  # 32
        self.bottleneck = ConvBlock(base_ch * 8, base_ch * 8, 2)  # 16

    def forward(self, x):
        f1 = self.e1(x)
        f2 = self.e2(f1)
        f3 = self.e3(f2)
        f4 = self.e4(f3)
        fb = self.bottleneck(f4)
        return fb, (f1, f2, f3, f4)


class DraftHead(nn.Module):
    def __init__(self, base_ch: int = 64):
        super().__init__()
        self.u4 = UpBlock(base_ch * 8, base_ch * 8, base_ch * 4)
        self.u3 = UpBlock(base_ch * 4, base_ch * 4, base_ch * 2)
        self.u2 = UpBlock(base_ch * 2, base_ch * 2, base_ch)
        self.u1 = UpBlock(base_ch, base_ch, base_ch)
        self.out = nn.Conv2d(base_ch, 3, kernel_size=3, padding=1)

    def forward(self, fb, skips):
        f1, f2, f3, f4 = skips
        x = self.u4(fb, f4)
        x = self.u3(x, f3)
        x = self.u2(x, f2)
        x = self.u1(x, f1)
        return torch.tanh(self.out(x))


class ImportanceHead(nn.Module):
    def __init__(self, base_ch: int = 64):
        super().__init__()
        self.u4 = UpBlock(base_ch * 8, base_ch * 8, base_ch * 4)
        self.u3 = UpBlock(base_ch * 4, base_ch * 4, base_ch * 2)
        self.u2 = UpBlock(base_ch * 2, base_ch * 2, base_ch)
        self.u1 = UpBlock(base_ch, base_ch, base_ch)
        self.out = nn.Conv2d(base_ch, 1, kernel_size=1)

    def forward(self, fb, skips):
        f1, f2, f3, f4 = skips
        x = self.u4(fb, f4)
        x = self.u3(x, f3)
        x = self.u2(x, f2)
        x = self.u1(x, f1)
        return torch.sigmoid(self.out(x))


class RefineHead(nn.Module):
    def __init__(self, in_ch: int = 8, base_ch: int = 64):
        super().__init__()
        # in_ch = masked_image(3) + draft(3) + mask(1) + importance(1)
        self.net = nn.Sequential(
            ConvBlock(in_ch, base_ch),
            ConvBlock(base_ch, base_ch),
            ConvBlock(base_ch, base_ch),
            nn.Conv2d(base_ch, 3, kernel_size=3, padding=1),
            nn.Tanh(),
        )

    def forward(self, x):
        return self.net(x)


class FrequencyAwareInpaintingNet(nn.Module):
    """
    가장 단순한 유효성 검증용 구조
    1) shared encoder
    2) draft coarse inpainting
    3) importance map prediction
    4) map-guided residual refinement
    """

    def __init__(self, base_ch: int = 64):
        super().__init__()
        self.encoder = SharedEncoder(in_ch=4, base_ch=base_ch)
        self.draft_head = DraftHead(base_ch=base_ch)
        self.importance_head = ImportanceHead(base_ch=base_ch)
        self.refine_head = RefineHead(in_ch=8, base_ch=base_ch)

    def forward(self, masked_img: torch.Tensor, mask: torch.Tensor):
        x = torch.cat([masked_img, mask], dim=1)
        fb, skips = self.encoder(x)

        draft = self.draft_head(fb, skips)
        importance = self.importance_head(fb, skips)

        refine_in = torch.cat([masked_img, draft, mask, importance], dim=1)
        residual = self.refine_head(refine_in)

        # 핵심 아이디어: importance가 높은 위치만 refinement를 강하게 반영
        refined = draft + importance * residual
        refined = refined.clamp(-1, 1)

        # known region은 원본 유지, hole만 예측 사용
        comp_draft = masked_img * (1.0 - mask) + draft * mask
        comp_refined = masked_img * (1.0 - mask) + refined * mask

        return {
            "draft": draft,
            "importance": importance,
            "residual": residual,
            "refined": refined,
            "comp_draft": comp_draft,
            "comp_refined": comp_refined,
        }


# ============================================================
# 6. Loss
# ============================================================


def wavelet_loss(pred: torch.Tensor, target: torch.Tensor, dwt: HaarDWT) -> torch.Tensor:
    pll, plh, phl, phh = dwt(pred)
    tll, tlh, thl, thh = dwt(target)
    return (
        F.l1_loss(pll, tll)
        + F.l1_loss(plh, tlh)
        + F.l1_loss(phl, thl)
        + F.l1_loss(phh, thh)
    )


def compute_losses(outputs, batch, dwt, cfg: Config):
    gt = batch["image"]
    mask = batch["mask"]

    pred_raw = outputs["refined"]         # composite 전
    pred_comp = outputs["comp_refined"]   # 시각화/평가용
    importance = outputs["importance"]

    target_map = build_importance_target(gt, dwt)

    rec = F.l1_loss(pred_raw, gt)
    masked = (mask * (pred_raw - gt).abs()).mean()
    map_loss = F.l1_loss(importance, target_map)
    wav = wavelet_loss(pred_raw, gt, dwt)
    sparse = importance.mean()

    total = (
        cfg.lambda_rec * rec
        + cfg.lambda_mask * masked
        + cfg.lambda_map * map_loss
        + cfg.lambda_wavelet * wav
        + cfg.lambda_sparse * sparse
    )

    logs = {
        "loss_total": total.item(),
        "loss_rec": rec.item(),
        "loss_mask": masked.item(),
        "loss_map": map_loss.item(),
        "loss_wav": wav.item(),
        "loss_sparse": sparse.item(),
    }
    return total, logs, target_map


# ============================================================
# 7. Visualization
# ============================================================


def denorm(x: torch.Tensor) -> torch.Tensor:
    return (x * 0.5 + 0.5).clamp(0, 1)


def save_visuals(batch, outputs, target_map, step: int, save_dir: str):
    img = batch["image"][:4]
    masked = batch["masked_image"][:4]
    mask = batch["mask"][:4].repeat(1, 3, 1, 1)
    draft = outputs["comp_draft"][:4]
    refined = outputs["comp_refined"][:4]
    importance = outputs["importance"][:4].repeat(1, 3, 1, 1)
    target_map = target_map[:4].repeat(1, 3, 1, 1)

    grid = torch.cat(
        [
            denorm(img),
            denorm(masked),
            mask,
            denorm(draft),
            denorm(refined),
            importance,
            target_map,
        ],
        dim=0,
    )
    save_path = os.path.join(save_dir, f"step_{step:07d}.png")
    save_image(grid, save_path, nrow=4)


# ============================================================
# 8. Train / Eval
# ============================================================


def train_one_epoch(model, loader, optimizer, dwt, device, epoch, cfg: Config, global_step: int):
    model.train()
    running = 0.0

    for batch in loader:
        for k in batch:
            batch[k] = batch[k].to(device)

        outputs = model(batch["masked_image"], batch["mask"])
        total, logs, target_map = compute_losses(outputs, batch, dwt, cfg)

        optimizer.zero_grad(set_to_none=True)
        total.backward()
        optimizer.step()

        running += logs["loss_total"]

        if global_step % cfg.log_interval == 0:
            print(
                f"[Train] epoch={epoch} step={global_step} "
                f"total={logs['loss_total']:.4f} "
                f"rec={logs['loss_rec']:.4f} "
                f"mask={logs['loss_mask']:.4f} "
                f"map={logs['loss_map']:.4f} "
                f"wav={logs['loss_wav']:.4f}"
            )
            save_visuals(batch, outputs, target_map, global_step, cfg.save_dir)

        global_step += 1

    return running / max(1, len(loader)), global_step


@torch.no_grad()
def validate(model, loader, dwt, device, cfg: Config):
    model.eval()
    losses = []
    for batch in loader:
        for k in batch:
            batch[k] = batch[k].to(device)
        outputs = model(batch["masked_image"], batch["mask"])
        total, _, _ = compute_losses(outputs, batch, dwt, cfg)
        losses.append(total.item())
    return sum(losses) / max(1, len(losses))


# ============================================================
# 9. Main
# ============================================================


def main():
    print(f"Using device: {CFG.device}")

    if not os.path.exists(CFG.data_root):
        raise FileNotFoundError(
            f"data_root={CFG.data_root} not found. "
            "ImageFolder 구조의 데이터셋 폴더를 준비하세요. 예: data/class_a/*.jpg"
        )

    full_dataset = InpaintingDataset(CFG.data_root, CFG.image_size)
    n_total = len(full_dataset)
    n_train = int(n_total * 0.9)
    n_val = n_total - n_train
    train_set, val_set = torch.utils.data.random_split(full_dataset, [n_train, n_val])

    train_loader = DataLoader(
        train_set,
        batch_size=CFG.batch_size,
        shuffle=True,
        num_workers=CFG.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=CFG.batch_size,
        shuffle=False,
        num_workers=CFG.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    model = FrequencyAwareInpaintingNet(base_ch=CFG.base_ch).to(CFG.device)
    dwt = HaarDWT().to(CFG.device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=CFG.lr, weight_decay=CFG.weight_decay
    )

    best_val = float("inf")
    global_step = 0

    for epoch in range(1, CFG.epochs + 1):
        train_loss, global_step = train_one_epoch(
            model, train_loader, optimizer, dwt, CFG.device, epoch, CFG, global_step
        )
        val_loss = validate(model, val_loader, dwt, CFG.device, CFG)
        print(f"[Epoch {epoch}] train={train_loss:.4f} val={val_loss:.4f}")

        ckpt = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "val_loss": val_loss,
            "config": CFG.__dict__,
        }
        torch.save(ckpt, os.path.join(CFG.save_dir, "last.pt"))

        if val_loss < best_val:
            best_val = val_loss
            torch.save(ckpt, os.path.join(CFG.save_dir, "best.pt"))
            print(f"Saved new best checkpoint: val={best_val:.4f}")


if __name__ == "__main__":
    main()
