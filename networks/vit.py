"""copyright (c) 2021 PaddlePaddle Authors. All Rights Reserve.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

# ===========================================================================================
# Feature-Proxy Transformer (FPTrans) in PaddlePaddle
#
# This code file is copied and modified from PaddleSeg
# for the development of FPTrans.
#
# By Jian-Wei Zhang (Github: Jarvis73).
# 2022-11
#
# ===========================================================================================

"""
import math
import logging
from functools import partial
from pathlib import Path
import numpy as np

import torch
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import paddle.nn.initializer as paddle_init

from networks import vit_utils

_logger = logging.getLogger(name=Path(__file__).parents[1].stem)

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)
IMAGENET_INCEPTION_MEAN = (0.5, 0.5, 0.5)
IMAGENET_INCEPTION_STD = (0.5, 0.5, 0.5)


class Attention(nn.Layer):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias_attr=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = paddle.shape(x)
        qkv = self.qkv(x).reshape((B, N, 3, self.num_heads, C // self.num_heads)).transpose((2, 0, 3, 1, 4))
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q.matmul(k.transpose((0, 1, 3, 2)))) * self.scale
        attn = F.softmax(attn, axis=-1)
        attn = self.attn_drop(attn)

        x = (attn.matmul(v)).transpose((0, 2, 1, 3)).reshape((B, N, C))
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Layer):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=None):
        super().__init__()
        norm_layer = norm_layer or partial(nn.LayerNorm, epsilon=1e-6)
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = vit_utils.DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = vit_utils.Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class VisionTransformer(nn.Layer):
    """ Vision Transformer
    A PyTorch impl of : `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale`
        - https://arxiv.org/abs/2010.11929
    Includes distillation token & head support for `DeiT: Data-efficient Image Transformers`
        - https://arxiv.org/abs/2012.12877
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 pretrained="",
                 num_heads=12, mlp_ratio=4., qkv_bias=True, representation_size=None, distilled=False,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., embed_layer=vit_utils.PatchEmbed, norm_layer=None,
                 act_layer=None, weight_init='',
                 opt=None, logger=None, original=False):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_chans (int): number of input channels
            num_classes (int): number of classes for classification head
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            representation_size (Optional[int]): enable and set representation layer (pre-logits) to this value if set
            distilled (bool): model includes a distillation token and head as in DeiT models
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            embed_layer (nn.Module): patch embedding layer
            norm_layer: (nn.Module): normalization layer
            weight_init: (str): weight init scheme
        """
        super().__init__()
        self.opt = opt
        self.logger = logger
        self.allow_mod = not original
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_tokens = 2 if distilled else 1
        norm_layer = norm_layer or partial(nn.LayerNorm, epsilon=1e-6)
        act_layer = act_layer or nn.GELU

        # Patch embedding
        self.patch_embed = embed_layer(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim, stride=opt.vit_stride)
        num_patches = self.patch_embed.num_patches

        # prompt tokens
        self.prompt_tokens = None
        if self.allow_mod:
            ncls = 15 if opt.dataset == "PASCAL" else 60
            divider = 1 + opt.bg_num * opt.shot
            self.prompt_tokens = self.create_parameter(
                [ncls * divider, opt.num_prompt // divider, embed_dim],
                default_initializer=nn.initializer.TruncatedNormal(std=opt.pt_std))
            self.sampler = np.random.RandomState(1234)

        # Class token, Distillation token
        self.cls_token = self.create_parameter(
            [1, 1, embed_dim], default_initializer=nn.initializer.Constant(0.))
        self.dist_token = self.create_parameter(
            [1, 1, embed_dim], default_initializer=nn.initializer.Constant(0.))\
            if distilled else None
        self.pos_embed = self.create_parameter(
            [1, num_patches + self.num_tokens, embed_dim],
            default_initializer=nn.initializer.TruncatedNormal(std=.02))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = np.linspace(0, drop_path_rate, depth)  # stochastic depth decay rule
        self.blocks = nn.Sequential(*[
            Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                act_layer=act_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        # Representation layer
        if representation_size and not distilled:
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(
                nn.Linear(embed_dim, representation_size, name='fc'),
                nn.Tanh(name='act')
            )
        else:
            self.pre_logits = nn.Identity()

        # Classifier head(s)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
        self.head_dist = None
        if distilled:
            self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if num_classes > 0 else nn.Identity()
        # for k, v in self.state_dict().items():
        #     print(k, v.shape)
        # input()

        if pretrained == "":
            self.init_weights(weight_init)
        else:
            if str(pretrained).endswith('.pth'):
                _load_weights_pth(logger, self, pretrained)
            elif str(pretrained).endswith('.npz'):
                _load_weights_npz(self, pretrained)
            else:
                raise ValueError(f'Not recognized file {pretrained}. [.pth|.npz]')

            if logger is not None:
                logger.info(' ' * 5 + f'==> {opt.backbone} initialized from {pretrained}')

    def init_weights(self, mode=''):
        assert mode in ('jax', 'jax_nlhb', 'nlhb', '')
        head_bias = -math.log(self.num_classes) if 'nlhb' in mode else 0.
        vit_utils.trunc_normal_(self.pos_embed, std=.02)
        if self.dist_token is not None:
            vit_utils.trunc_normal_(self.dist_token, std=.02)
        if mode.startswith('jax'):
            # leave cls token as zeros to match jax impl
            vit_utils.named_apply(partial(_init_vit_weights, head_bias=head_bias, jax_impl=True), self)
        else:
            vit_utils.trunc_normal_(self.cls_token, std=.02)
            self.apply(_init_vit_weights)

    def forward_original(self, x):
        # x: [B, C, H, W]
        x = self.patch_embed(x)  # [B, N ,C]
        cls_token = expand_to_batch(self.cls_token, x.shape[0])  # [B, 1, C]
        if self.dist_token is None:
            x = cat_token(cls_token, x)     # [B, N+1, C]
        else:
            dist_token = expand_to_batch(self.dist_token, x.shape[0])  # [B, 1, C]
            x = cat_token(cls_token, dist_token, x)     # [B, N+2, C]
        x = x + self.pos_embed  # [B, N + 1, C]
        x = self.pos_drop(x)

        for i, block in enumerate(self.blocks):
            x = block(x)
        x = self.norm(x)

        x = x[:, self.num_tokens:, :]
        b, n, c = x.shape
        hh = int(math.sqrt(n))
        x = x.reshape((b, hh, hh, c)).transpose((0, 3, 1, 2))

        return dict(out=x)

    def forward(self, x):
        if not self.allow_mod:
            return self.forward_original(x)

        # x: [B, C, H, W]
        # fg_token: [B, 1, C]
        # bg_token: [B, k, C]
        x, (fg_token, bg_token) = x
        bank_size, psize, embed_dim = self.prompt_tokens.shape
        B = x.shape[0] // (1 + self.opt.shot)
        divider = 1 + self.opt.bg_num * self.opt.shot
        prompts = self.prompt_tokens[self.sampler.choice(bank_size, size=B * divider, replace=False)] \
            .reshape((B, divider * psize, embed_dim))    # [B, 2*psize, embed_dim]
        tokens = {
            'fg': prompts[:, :psize] + fg_token,
            'bg': prompts[:, psize:] + bg_token.unsqueeze(2).expand((-1, -1, psize, -1)).reshape(
                (B, (divider - 1) * psize, embed_dim))
        }

        x = self.patch_embed(x)  # [B, N ,C]
        cls_token = expand_to_batch(self.cls_token, x.shape[0])  # [B, 1, C]
        if self.dist_token is None:
            x = cat_token(cls_token, x)     # [B, N+1, C]
        else:
            dist_token = expand_to_batch(self.dist_token, x.shape[0])  # [B, 1, C]
            x = cat_token(cls_token, dist_token, x)     # [B, N+2, C]
        x = x + self.pos_embed  # [B, N + 1, C']
        x = self.pos_drop(x)

        S = self.opt.shot
        B = x.shape[0] // (S + 1)

        # Concat image and tokens
        _, N, C = x.shape
        n1 = tokens['fg'].shape[1]
        n2 = tokens['bg'].shape[1]
        fg_token = tokens['fg'].reshape((B, 1, n1, C)).expand((-1, S + 1, -1, -1)).reshape((B*(S+1), n1, C))
        bg_token = tokens['bg'].reshape((B, 1, n2, C)).expand((-1, S + 1, -1, -1)).reshape((B*(S+1), n2, C))
        x = cat_token(x, fg_token, bg_token)

        # Forward transformer blocks
        for i, block in enumerate(self.blocks):
            x = block(x)
            x = self.reduce_and_expand(x, num=S + 1, start=self.num_tokens, end=n1 + n2)

        # Split outputs
        x, fg_token, bg_token = x.split([N, n1, n2], axis=1)
        tokens['fg'] = fg_token.reshape((B, S + 1, n1, C))[:, 0]
        tokens['bg'] = bg_token.reshape((B, S + 1, n2, C))[:, 0]

        x = self.norm(x)

        x = x[:, self.num_tokens:, :]
        b, n, c = x.shape
        hh = int(math.sqrt(n))
        x = x.reshape((b, hh, hh, c)).transpose((0, 3, 1, 2))

        tokens['fg'] = tokens['fg'].mean(1)[:, :, None, None]    # [B, C, 1, 1]
        if self.opt.bg_num == 1:
            tokens['bg'] = tokens['bg'].mean(1)[:, :, None, None]    # [B, C, 1, 1]
        else:
            _B, _N, _C = tokens['bg'].shape
            bg_num = self.opt.bg_num
            tokens['bg'] = tokens['bg'].reshape(
                (_B * bg_num, _N // bg_num, _C)).mean(1)[:, :, None, None]    # [Bk, C, 1, 1]

        return dict(out=x, tokens=tokens)

    @staticmethod
    def reduce_and_expand(x, num, start=1, end=0):
        B, N, C = x.shape
        x = x.reshape((B // num, num, N, C))
        if start > 0:
            mean_token = x[:, :, :start, :].mean(1, keepdim=True).expand((-1, num, -1, -1))
            x[:, :, :start, :] = mean_token
        if end > 0:
            mean_token = x[:, :, -end:, :].mean(1, keepdim=True).expand((-1, num, -1, -1))
            x[:, :, -end:, :] = mean_token
        return x.reshape((B, N, C))


def expand_to_batch(prompt, batch_size, stack=False):
    if stack:
        prompt = paddle.unsqueeze(prompt, 0)
    return prompt.expand((batch_size, *[-1 for _ in prompt.shape[1:]]))


def cat_token(*args):
    return paddle.concat(args, axis=1)


def _init_vit_weights(module: nn.Layer, name: str = '', head_bias: float = 0., jax_impl: bool = False):
    """ ViT weight initialization
    * When called without n, head_bias, jax_impl args it will behave exactly the same
      as my original init for compatibility with prev hparam / downstream use cases (ie DeiT).
    * When called w/ valid n (module name) and jax_impl=True, will (hopefully) match JAX impl
    """
    if isinstance(module, nn.Linear):
        if name.startswith('head'):
            vit_utils.zeros_(module.weight)
            vit_utils.init_wraper(paddle_init.Constant(head_bias), module.bias)
        elif name.startswith('pre_logits'):
            vit_utils.lecun_normal_(module.weight)
            vit_utils.zeros_(module.bias)
        else:
            if jax_impl:
                vit_utils.init_wraper(paddle_init.XavierUniform(), module.weight)
                if module.bias is not None:
                    if 'mlp' in name:
                        vit_utils.init_wraper(paddle_init.Normal(std=1e-6), module.bias)
                    else:
                        vit_utils.zeros_(module.bias)
            else:
                vit_utils.trunc_normal_(module.weight)
                if module.bias is not None:
                    vit_utils.zeros_(module.bias)
    elif jax_impl and isinstance(module, nn.Conv2D):
        # NOTE conv was left to pytorch default in my original init
        vit_utils.lecun_normal_(module.weight)
        if module.bias is not None:
            vit_utils.zeros_(module.bias)
    elif isinstance(module, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm2D)):
        vit_utils.zeros_(module.bias)
        vit_utils.ones_(module.weight)


@paddle.no_grad()
def _load_weights_npz(model: VisionTransformer, checkpoint_path: str, prefix: str = ''):
    """ Load weights from .npz checkpoints for official Google Brain Flax implementation
    """
    def _n2p(w, t=True):
        if w.ndim == 4 and w.shape[0] == w.shape[1] == w.shape[2] == 1:
            w = w.flatten()
        if t:
            if w.ndim == 4:
                w = w.transpose([3, 2, 0, 1])
            elif w.ndim == 3:
                w = w.transpose([2, 0, 1])
            elif w.ndim == 2:
                # w = w.transpose([1, 0])
                pass    # paddle.nn.Linear.weight has the right shape (in_c, out_c)
        return paddle.to_tensor(w)

    w = np.load(checkpoint_path)
    if not prefix and 'opt/target/embedding/kernel' in w:
        prefix = 'opt/target/'

    if hasattr(model.patch_embed, 'backbone'):
        # hybrid
        backbone = model.patch_embed.backbone
        stem_only = not hasattr(backbone, 'stem')
        stem = backbone if stem_only else backbone.stem
        stem.conv.weight.set_value(vit_utils.adapt_input_conv(stem.conv.weight.shape[1], _n2p(w[f'{prefix}conv_root/kernel'])))
        stem.norm.weight.set_value(_n2p(w[f'{prefix}gn_root/scale']))
        stem.norm.bias.set_value(_n2p(w[f'{prefix}gn_root/bias']))
        if not stem_only:
            for i, stage in enumerate(backbone.stages):
                for j, block in enumerate(stage.blocks):
                    bp = f'{prefix}block{i + 1}/unit{j + 1}/'
                    for r in range(3):
                        getattr(block, f'conv{r + 1}').weight.set_value(_n2p(w[f'{bp}conv{r + 1}/kernel']))
                        getattr(block, f'norm{r + 1}').weight.set_value(_n2p(w[f'{bp}gn{r + 1}/scale']))
                        getattr(block, f'norm{r + 1}').bias.set_value(_n2p(w[f'{bp}gn{r + 1}/bias']))
                    if block.downsample is not None:
                        block.downsample.conv.weight.set_value(_n2p(w[f'{bp}conv_proj/kernel']))
                        block.downsample.norm.weight.set_value(_n2p(w[f'{bp}gn_proj/scale']))
                        block.downsample.norm.bias.set_value(_n2p(w[f'{bp}gn_proj/bias']))
        embed_conv_w = _n2p(w[f'{prefix}embedding/kernel'])
    else:
        embed_conv_w = vit_utils.adapt_input_conv(
            model.patch_embed.proj.weight.shape[1], _n2p(w[f'{prefix}embedding/kernel']))
    model.patch_embed.proj.weight.set_value(embed_conv_w)
    model.patch_embed.proj.bias.set_value(_n2p(w[f'{prefix}embedding/bias']))
    model.cls_token.set_value(_n2p(w[f'{prefix}cls'], t=False))
    pos_embed_w = _n2p(w[f'{prefix}Transformer/posembed_input/pos_embedding'], t=False)
    if pos_embed_w.shape != model.pos_embed.shape:
        pos_embed_w = resize_pos_embed(  # resize pos embedding when different size from pretrained weights
            pos_embed_w, model.pos_embed, getattr(model, 'num_tokens', 1), model.patch_embed.grid_size)
    model.pos_embed.set_value(pos_embed_w)
    model.norm.weight.set_value(_n2p(w[f'{prefix}Transformer/encoder_norm/scale']))
    model.norm.bias.set_value(_n2p(w[f'{prefix}Transformer/encoder_norm/bias']))
    if isinstance(model.head, nn.Linear) and model.head.bias.shape[0] == w[f'{prefix}head/bias'].shape[-1]:
        model.head.weight.set_value(_n2p(w[f'{prefix}head/kernel'], t=False))   # paddle, t=False
        model.head.bias.set_value(_n2p(w[f'{prefix}head/bias']))
    if isinstance(getattr(model.pre_logits, 'fc', None), nn.Linear) and f'{prefix}pre_logits/bias' in w:
        model.pre_logits.fc.weight.set_value(_n2p(w[f'{prefix}pre_logits/kernel'], t=False))    # paddle, t=False
        model.pre_logits.fc.bias.set_value(_n2p(w[f'{prefix}pre_logits/bias']))
    for i, block in enumerate(model.blocks.children()):
        block_prefix = f'{prefix}Transformer/encoderblock_{i}/'
        mha_prefix = block_prefix + 'MultiHeadDotProductAttention_1/'
        block.norm1.weight.set_value(_n2p(w[f'{block_prefix}LayerNorm_0/scale']))
        block.norm1.bias.set_value(_n2p(w[f'{block_prefix}LayerNorm_0/bias']))
        block.attn.qkv.weight.set_value(paddle.concat([     # paddle, no .T after flatten(1) and then concat along axis=1
            _n2p(w[f'{mha_prefix}{n}/kernel'], t=False).flatten(1) for n in ('query', 'key', 'value')], axis=1))
        block.attn.qkv.bias.set_value(paddle.concat([
            _n2p(w[f'{mha_prefix}{n}/bias'], t=False).reshape((-1,)) for n in ('query', 'key', 'value')]))
        block.attn.proj.weight.set_value(_n2p(w[f'{mha_prefix}out/kernel']).flatten(1).t())     # paddle, use t()
        block.attn.proj.bias.set_value(_n2p(w[f'{mha_prefix}out/bias']))
        for r in range(2):
            getattr(block.mlp, f'fc{r + 1}').weight.set_value(_n2p(w[f'{block_prefix}MlpBlock_3/Dense_{r}/kernel'], t=False))   # paddle, t=False
            getattr(block.mlp, f'fc{r + 1}').bias.set_value(_n2p(w[f'{block_prefix}MlpBlock_3/Dense_{r}/bias']))
        block.norm2.weight.set_value(_n2p(w[f'{block_prefix}LayerNorm_2/scale']))
        block.norm2.bias.set_value(_n2p(w[f'{block_prefix}LayerNorm_2/bias']))


@paddle.no_grad()
def _load_weights_pth(logger, model: VisionTransformer, checkpoint_path):
    ckpt = torch.load(checkpoint_path, map_location='cpu')['model']
    # torch.Tensor --> paddle.Tensor
    ckpt = {k: paddle.to_tensor(v.numpy()) for k, v in ckpt.items()}
    state_dict = model.state_dict()
    if ckpt['pos_embed'].shape != state_dict['pos_embed'].shape:
        ckpt['pos_embed'] = resize_pos_embed(
            ckpt['pos_embed'], state_dict['pos_embed'], model.num_tokens, model.patch_embed.grid_size)

    counter = 0
    for k in state_dict.keys():
        if k in ckpt:
            state_dict[k] = ckpt[k]
            counter += 1

    logger.info(' ' * 5 + f"==> {counter} parameters loaded.")
    model.set_state_dict(state_dict)


def resize_pos_embed(posemb, posemb_new, num_tokens=1, gs_new=()):
    # Rescale the grid of position embeddings when loading from state_dict. Adapted from
    # https://github.com/google-research/vision_transformer/blob/00883dd691c63a6830751563748663526e811cee/vit_jax/checkpoint.py#L224
    _logger.info(' ' * 5 + '==> Resized position embedding: %s to %s', posemb.shape, posemb_new.shape)
    ntok_new = posemb_new.shape[1]
    if num_tokens:
        posemb_tok, posemb_grid = posemb[:, :num_tokens], posemb[0, num_tokens:]
        ntok_new -= num_tokens
    else:
        posemb_tok, posemb_grid = posemb[:, :0], posemb[0]
    gs_old = int(math.sqrt(len(posemb_grid)))
    if not len(gs_new):  # backwards compatibility
        gs_new = [int(math.sqrt(ntok_new))] * 2
    assert len(gs_new) >= 2
    _logger.info(' ' * 5 + '==> Position embedding grid-size from %s to %s', [gs_old, gs_old], gs_new)
    posemb_grid = posemb_grid.reshape((1, gs_old, gs_old, -1)).transpose((0, 3, 1, 2))
    posemb_grid = F.interpolate(posemb_grid, size=gs_new, mode='bicubic', align_corners=False)
    posemb_grid = posemb_grid.transpose((0, 2, 3, 1)).reshape((1, gs_new[0] * gs_new[1], -1))
    posemb = paddle.concat([posemb_tok, posemb_grid], axis=1)
    return posemb


def checkpoint_filter_fn(state_dict, model):
    """ convert patch embedding weight from manual patchify + linear proj to conv"""
    out_dict = {}
    if 'model' in state_dict:
        # For deit models
        state_dict = state_dict['model']
    for k, v in state_dict.items():
        if 'patch_embed.proj.weight' in k and len(v.shape) < 4:
            # For old models that I trained prior to conv based patchification
            O, I, H, W = model.patch_embed.proj.weight.shape
            v = v.reshape((O, -1, H, W))
        elif k == 'pos_embed' and v.shape != model.pos_embed.shape:
            # To resize pos embedding when using model at different size from pretrained weights
            v = resize_pos_embed(
                v, model.pos_embed, getattr(model, 'num_tokens', 1), model.patch_embed.grid_size)
        out_dict[k] = v
    return out_dict


vit_factory = {
    'ViT-Ti/16':         {'patch_size': 16, 'embed_dim':  192, 'depth': 12, 'num_heads':  3, 'distilled': False},
    'ViT-S/32':          {'patch_size': 32, 'embed_dim':  384, 'depth': 12, 'num_heads':  6, 'distilled': False},
    'ViT-S/16':          {'patch_size': 16, 'embed_dim':  384, 'depth': 12, 'num_heads':  6, 'distilled': False},
    'ViT-S/16-i21k':     {'patch_size': 16, 'embed_dim':  384, 'depth': 12, 'num_heads':  6, 'distilled': False},
    'ViT-B/32':          {'patch_size': 32, 'embed_dim':  768, 'depth': 12, 'num_heads': 12, 'distilled': False},
    'ViT-B/16':          {'patch_size': 16, 'embed_dim':  768, 'depth': 12, 'num_heads': 12, 'distilled': False},
    'ViT-B/16-384':      {'patch_size': 16, 'embed_dim':  768, 'depth': 12, 'num_heads': 12, 'distilled': False},
    'ViT-B/16-i21k':     {'patch_size': 16, 'embed_dim':  768, 'depth': 12, 'num_heads': 12, 'distilled': False},
    'ViT-B/16-i21k-384': {'patch_size': 16, 'embed_dim':  768, 'depth': 12, 'num_heads': 12, 'distilled': False},
    'ViT-B/8':           {'patch_size':  8, 'embed_dim':  768, 'depth': 12, 'num_heads': 12, 'distilled': False},
    'ViT-L/32':          {'patch_size': 32, 'embed_dim': 1024, 'depth': 24, 'num_heads': 16, 'distilled': False},
    'ViT-L/16':          {'patch_size': 16, 'embed_dim': 1024, 'depth': 24, 'num_heads': 16, 'distilled': False},
    'ViT-L/16-384':      {'patch_size': 16, 'embed_dim': 1024, 'depth': 24, 'num_heads': 16, 'distilled': False},

    'DeiT-T/16':         {'patch_size': 16, 'embed_dim':  192, 'depth': 12, 'num_heads': 3, 'distilled': True},
    'DeiT-S/16':         {'patch_size': 16, 'embed_dim':  384, 'depth': 12, 'num_heads': 6, 'distilled': True},
    'DeiT-B/16':         {'patch_size': 16, 'embed_dim':  768, 'depth': 12, 'num_heads': 12, 'distilled': True},
    'DeiT-B/16-384':     {'patch_size': 16, 'embed_dim':  768, 'depth': 12, 'num_heads': 12, 'distilled': True},
}


def vit_model(model_type,
              image_size,
              pretrained="",
              init_channels=3,
              num_classes=1000,
              opt=None,
              logger=None,
              original=False,
              depth=None):
    return VisionTransformer(img_size=image_size,
                             patch_size=vit_factory[model_type]['patch_size'],
                             in_chans=init_channels,
                             num_classes=num_classes,
                             embed_dim=vit_factory[model_type]['embed_dim'],
                             depth=depth or opt.vit_depth or vit_factory[model_type]['depth'],
                             num_heads=vit_factory[model_type]['num_heads'],
                             pretrained=pretrained,
                             distilled=vit_factory[model_type]['distilled'],
                             opt=opt,
                             logger=logger,
                             original=original)


