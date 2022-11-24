import sys
from pathlib import Path

import numpy as np
import torch
import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from networks.layers import DropBlock2D
from constants import pretrained_weights, model_urls
from core.losses import get as get_loss
from networks import vit
from utils_.misc import interpb, interpn


class Residual(nn.Layer):
    def __init__(self, layers, up=2):
        super().__init__()
        self.layers = layers
        self.up = up

    def forward(self, x):
        h, w = x.shape[-2:]
        x_up = interpb(x, (h * self.up, w * self.up))
        x = x_up + self.layers(x)
        return x


class FPTrans(nn.Layer):
    def __init__(self, opt, logger):
        super(FPTrans, self).__init__()
        self.opt = opt
        self.logger = logger
        self.shot = opt.shot
        self.drop_dim = opt.drop_dim
        self.drop_rate = opt.drop_rate
        self.drop2d_kwargs = {'drop_prob': opt.drop_rate, 'block_size': opt.block_size}

        # Check existence.
        pretrained = self.get_or_download_pretrained(opt.backbone, opt.tqdm)

        # Main model
        self.encoder = nn.Sequential(
            ('backbone', vit.vit_model(opt.backbone,
                                       opt.height,
                                       pretrained=pretrained,
                                       num_classes=0,
                                       opt=opt,
                                       logger=logger))
        )
        embed_dim = vit.vit_factory[opt.backbone]['embed_dim']
        self.purifier = self.build_upsampler(embed_dim)
        self.__class__.__name__ = f"FPTrans/{opt.backbone}"

        # Pretrained model
        self.original_encoder = vit.vit_model(opt.backbone,
                                              opt.height,
                                              pretrained=pretrained,
                                              num_classes=0,
                                              opt=opt,
                                              logger=logger,
                                              original=True)
        for var in self.original_encoder.parameters():
            var.requires_grad = False

        # Define pair-wise loss
        self.pairwise_loss = get_loss(opt, logger, loss='pairwise')
        # Background sampler
        self.bg_sampler = np.random.RandomState(1289)

        logger.info(' ' * 5 + f"==> Model {self.__class__.__name__} created")

    def build_upsampler(self, embed_dim):
        return Residual(nn.Sequential(
            nn.Conv2D(embed_dim, 256, kernel_size=1),
            nn.ReLU(),
            nn.Dropout(self.drop_rate) if self.drop_dim == 1 else DropBlock2D(**self.drop2d_kwargs),
            nn.Conv2DTranspose(256, 256, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.Dropout(self.drop_rate) if self.drop_dim == 1 else DropBlock2D(**self.drop2d_kwargs),
            nn.Conv2D(256, embed_dim, kernel_size=1),
        ))

    def forward(self, x, s_x, s_y, y=None, out_shape=None):
        """

        Parameters
        ----------
        x: paddle.Tensor
            [B, C, H, W], query image
        s_x: paddle.Tensor
            [B, S, C, H, W], support image
        s_y: paddle.Tensor
            [B, S, H, W], support mask
        y: paddle.Tensor
            [B, 1, H, W], query mask, used for calculating the pair-wise loss
        out_shape: list
            The shape of the output predictions. If not provided, it is default
            to the last two dimensions of `y`. If `y` is also not provided, it is
            default to the [opt.height, opt.width].

        Returns
        -------
        output: dict
            'out': paddle.Tensor
                logits that predicted by feature proxies
            'out_prompt': paddle.Tensor
                logits that predicted by prompt proxies
            'loss_pair': float
                pair-wise loss
        """
        B, S, C, H, W = s_x.shape
        img_cat = paddle.concat((s_x, x.reshape((B, 1, C, H, W))), axis=1).reshape((B*(S+1), C, H, W))

        # Calculate class-aware prompts
        with paddle.no_grad():
            inp = s_x.reshape((B * S, C, H, W))
            # Forward
            sup_feat = self.original_encoder(inp)['out']
            _, c, h0, w0 = sup_feat.shape
            sup_mask = interpn(s_y.reshape((B*S, 1, H, W)), (h0, w0))                                # [BS, 1, h0, w0]
            sup_mask_fg = (sup_mask == 1).cast(paddle.float32)                                     # [BS, 1, h0, w0]
            # Calculate fg and bg tokens
            fg_token = (sup_feat * sup_mask_fg).sum((2, 3)) / (sup_mask_fg.sum((2, 3)) + 1e-6)
            fg_token = fg_token.reshape((B, S, c)).mean(1, keepdim=True)  # [B, 1, c]
            bg_token = self.compute_multiple_prototypes(
                self.opt.bg_num,
                sup_feat.reshape((B, S, c, h0, w0)),
                sup_mask == 0,
                self.bg_sampler
            ).transpose((0, 2, 1))    # [B, k, c]

        # Forward
        img_cat = (img_cat, (fg_token, bg_token))
        backbone_out = self.encoder(img_cat)

        features = self.purifier(backbone_out['out'])               # [B(S+1), c, h, w]
        _, c, h, w = features.shape
        features = features.reshape((B, S+1, c, h, w))                   # [B, S+1, c, h, w]
        sup_fts, qry_fts = features.split([S, 1], axis=1)            # [B, S, c, h, w] / [B, 1, c, h, w]
        sup_mask = interpn(s_y.reshape((B * S, 1, H, W)), (h, w))        # [BS, 1, h, w]

        pred = self.classifier(sup_fts, qry_fts, sup_mask)          # [B, 2, h, w]

        # Output
        if not out_shape:
            out_shape = y.shape[-2:] if y is not None else (H, W)
        out = interpb(pred, out_shape)    # [BQ, 2, *, *]
        output = dict(out=out)

        if self.training and y is not None:
            # Pairwise loss
            x1 = sup_fts.flatten(3)                 # [B, S, C, N]
            y1 = sup_mask.reshape((B, S, -1)).cast(paddle.int64)     # [B, S, N]
            x2 = qry_fts.flatten(3)                 # [B, 1, C, N]
            y2 = interpn(y.cast(paddle.float32), (h, w)).flatten(2).cast(paddle.int64)   # [B, 1, N]
            output['loss_pair'] = self.pairwise_loss(x1, y1, x2, y2)

            # Prompt-Proxy prediction
            fg_token = self.purifier(backbone_out['tokens']['fg'])[:, :, 0, 0]        # [B, c]
            bg_token = self.purifier(backbone_out['tokens']['bg'])[:, :, 0, 0]        # [B, c]
            bg_token = bg_token.reshape((B, self.opt.bg_num, c)).transpose((0, 2, 1))     # [B, c, k]
            pred_prompt = self.compute_similarity(fg_token, bg_token, qry_fts.reshape((-1, c, h, w)))

            # Up-sampling
            pred_prompt = interpb(pred_prompt, (H, W))
            output['out_prompt'] = pred_prompt

        return output

    def classifier(self, sup_fts, qry_fts, sup_mask):
        """

        Parameters
        ----------
        sup_fts: paddle.Tensor
            [B, S, c, h, w]
        qry_fts: paddle.Tensor
            [B, 1, c, h, w]
        sup_mask: paddle.Tensor
            [BS, 1, h, w]

        Returns
        -------
        pred: paddle.Tensor
            [B, 2, h, w]

        """
        B, S, c, h, w = sup_fts.shape

        # FG proxies
        sup_fg = (sup_mask == 1).reshape((-1, 1, h * w)).cast(paddle.float32)  # [BS, 1, hw]
        fg_vecs = paddle.sum(sup_fts.reshape((-1, c, h * w)) * sup_fg, axis=-1) / (sup_fg.sum(axis=-1) + 1e-5)     # [BS, c]
        # Merge multiple shots
        fg_proto = fg_vecs.reshape((B, S, c)).mean(axis=1)    # [B, c]

        # BG proxies
        bg_proto = self.compute_multiple_prototypes(self.opt.bg_num, sup_fts, sup_mask == 0, self.bg_sampler)

        # Calculate cosine similarity
        qry_fts = qry_fts.reshape((-1, c, h, w))
        pred = self.compute_similarity(fg_proto, bg_proto, qry_fts)   # [B, 2, h, w]
        return pred

    @staticmethod
    def compute_multiple_prototypes(bg_num, sup_fts, sup_bg, sampler):
        """

        Parameters
        ----------
        bg_num: int
            Background partition numbers
        sup_fts: paddle.Tensor
            [B, S, c, h, w]
        sup_bg: paddle.Tensor
            [BS, 1, h, w]
        sampler: np.random.RandomState

        Returns
        -------
        bg_proto: torch.Tensor
            [B, c, k], where k is the number of background proxies

        """
        B, S, c, h, w = sup_fts.shape
        bg_mask = sup_bg.reshape((B, S, h, w))    # [B, S, h, w]
        batch_bg_protos = []

        for b in range(B):
            bg_protos = []
            for s in range(S):
                bg_mask_i = bg_mask[b, s]     # [h, w]

                # Check if zero
                with paddle.no_grad():
                    if bg_mask_i.sum() < bg_num:
                        bg_mask_i = bg_mask[b, s].clone()    # don't change original mask
                        bg_mask_i.reshape((-1,))[:bg_num] = True

                # Iteratively select farthest points as centers of background local regions
                all_centers = []
                first = True
                pts = paddle.concat(paddle.where(bg_mask_i), axis=1)     # [N, 2]
                for _ in range(bg_num):
                    if first:
                        i = sampler.choice(pts.shape[0])
                        first = False
                    else:
                        dist = pts.reshape((-1, 1, 2)) - paddle.stack(all_centers, axis=0).reshape((1, -1, 2))
                        # choose the farthest point
                        i = paddle.argmax((dist ** 2).sum(-1).min(1))
                    pt = pts[i]   # center y, x
                    all_centers.append(pt)

                # Assign bg labels for bg pixels
                dist = pts.reshape((-1, 1, 2)) - paddle.stack(all_centers, axis=0).reshape((1, -1, 2))
                bg_labels = paddle.argmin((dist ** 2).sum(-1), axis=1)

                # Compute bg prototypes
                bg_feats = sup_fts[b, s].transpose((1, 2, 0))[bg_mask_i]    # [N, c]
                for i in range(bg_num):
                    proto = bg_feats[bg_labels == i].mean(0)    # [c]
                    bg_protos.append(proto)

            bg_protos = paddle.stack(bg_protos, axis=1)   # [c, k]
            batch_bg_protos.append(bg_protos)
        bg_proto = paddle.stack(batch_bg_protos, axis=0)  # [B, c, k]
        return bg_proto

    @staticmethod
    def compute_similarity(fg_proto, bg_proto, qry_fts, dist_scalar=20):
        """
        Parameters
        ----------
        fg_proto: torch.Tensor
            [B, c], foreground prototype
        bg_proto: torch.Tensor
            [B, c, k], multiple background prototypes
        qry_fts: torch.Tensor
            [B, c, h, w], query features
        dist_scalar: int
            scale factor on the results of cosine similarity

        Returns
        -------
        pred: torch.Tensor
            [B, 2, h, w], predictions
        """
        fg_distance = F.cosine_similarity(
            qry_fts, fg_proto[..., None, None], axis=1) * dist_scalar        # [B, h, w]
        if len(bg_proto.shape) == 3:    # multiple background protos
            bg_distances = []
            for i in range(bg_proto.shape[-1]):
                bg_p = bg_proto[:, :, i]
                bg_d = F.cosine_similarity(
                    qry_fts, bg_p[..., None, None], axis=1) * dist_scalar        # [B, h, w]
                bg_distances.append(bg_d)
            bg_distance = paddle.stack(bg_distances, axis=0).max(0)
        else:   # single background proto
            bg_distance = F.cosine_similarity(
                qry_fts, bg_proto[..., None, None], axis=1) * dist_scalar        # [B, h, w]
        pred = paddle.stack((bg_distance, fg_distance), axis=1)               # [B, 2, h, w]

        return pred

    def load_weights(self, ckpt_path, logger, strict=True):
        """
        Support load PaddlePaddle weights and PyTorch weights

        Parameters
        ----------
        ckpt_path: Path
            path to the checkpoint
        logger
        strict: bool
            strict mode or not

        """
        if ckpt_path.suffix == '.pth':
            weights = torch.load(str(ckpt_path), map_location='cpu')['model_state']
            # torch.Tensor --> paddle.Tensor
            weights = {k: paddle.to_tensor(v.numpy()) for k, v in weights.items()}
            # paddle Linear weight == torch Linear weight.T
            for k, v in weights.items():
                if 'attn.qkv.weight' in k or 'attn.proj.weight' in k \
                        or 'mlp.fc1.weight' in k or 'mlp.fc2.weight' in k:
                    weights[k] = v.t()
        elif ckpt_path.suffix == '.pdparams':
            weights = paddle.load(str(ckpt_path))
        else:
            raise ValueError(f'Unsupported checkpoint file suffix: {ckpt_path.suffix}. [.pth|.pdparams]')
        # Update with original_encoder
        weights.update({k: v for k, v in self.state_dict().items() if 'original_encoder' in k})

        self.set_state_dict(weights)
        logger.info(' ' * 5 + f"==> Model {self.__class__.__name__} initialized from {ckpt_path}")

    @staticmethod
    def get_or_download_pretrained(backbone, progress):
        if backbone not in pretrained_weights:
            raise ValueError(f'Not supported backbone {backbone}. '
                             f'Available backbones: {list(pretrained_weights.keys())}')

        cached_file = Path(pretrained_weights[backbone])
        if cached_file.exists():
            return cached_file

        # Try to download
        from torch.hub import download_url_to_file

        url = model_urls[backbone]
        cached_file.parent.mkdir(parents=True, exist_ok=True)
        sys.stderr.write('Downloading: "{}" to {}\n'.format(url, cached_file))
        download_url_to_file(url, str(cached_file), progress=progress)

        return cached_file

    def get_params_list(self):
        params = []
        for var in self.parameters():
            if not var.stop_gradient:
                params.append(var)
        return [{'params': params}]
