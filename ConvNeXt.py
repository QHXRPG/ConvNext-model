import torch
import torch.nn as nn

"""层正则化"""
class LayerNorm(nn.Module):
    def __init__(self, dim: int, norm_type: str, eps :float = 1e-6):
        super(LayerNorm, self).__init__()
        self.w = nn.Parameter(torch.ones(dim), requires_grad=True)
        self.b = nn.Parameter(torch.zeros(dim), requires_grad=True)
        self.norm_type = norm_type
        self.dim = dim
        self.eps = eps
        if self.norm_type!='first channel' or self.norm_type!='last channel':
            raise ValueError(f"找不到名为'{self.norm_type}'的正则化风格")

    def forward(self, x:torch.tensor):
        if self.norm_type == 'last channel':
            return nn.functional.layer_norm(input=x,
                                            normalized_shape=(self.dim,),
                                            weight=self.w,
                                            bias=self.b,
                                            eps=self.eps)
        else:
            mean = x.mean(1,keepdim=True)
            var = (x-mean).pow(2).mean(1,keepdim=True)
            x = ((x-mean)/torch.sqrt(var+self.eps))*self.w[:,None,None] + self.b[:,None,None]
            return x

"""DropPath正则化"""
class DropPath(nn.Module):
    def __init__(self, drop_value:float = 0.0, train:bool = True):
        super(DropPath, self).__init__()
        self.drop_value = drop_value
        self.train = train
    def drop_path(self, x: torch.tensor):
        if self.drop_value == 0 or not self.train:
            return x
        else:
            keep_value = 1-self.drop_value
            shape = (x.shape[0],) + (1,) * (x.ndim-1)
            random_tensor = keep_value + torch.rand(shape, dtype=x.dtype)
            random_tensor.floor_()
            output = x.div(keep_value) * random_tensor
            return output
    def forward(self, x:torch.tensor):
        return self.drop_path(x)

"""ConvNeXt_block"""
class ConvNeXt_block(nn.Module):
    def __init__(self, dim:int, scale_value:float, drop_value:float=.0):
        super(ConvNeXt_block, self).__init__()
        self.depthwise_conv = nn.Conv2d(in_channels=dim, out_channels=dim,
                                        kernel_size=7, stride=1, padding=3)
        self.norm = LayerNorm(dim=dim, norm_type="last channel", eps=1e-6)
        self.conv1 = nn.Conv2d(in_channels=dim, out_channels=4*dim,
                               kernel_size=1, stride=1, padding=1)
        self.gelu = nn.GELU()
        self.conv2 = nn.Conv2d(in_channels=4*dim, out_channels=dim,
                               kernel_size=1, stride=1, padding=1)
        self.gama = nn.Parameter(torch.ones((dim,))*scale_value, requires_grad=True) \
            if scale_value > 0 else None
        self.drop_path = DropPath(drop_value) if drop_value>0. else nn.Identity()
    def forward(self, x:torch.tensor):
        shortcut = x
        x = self.depthwise_conv(x)
        x = x.permute(0, 2, 3, 1)  # [N, C, H, W] -> [N, H, W, C]
        x = self.norm(x)
        x = self.conv1(x)
        x = self.gelu(x)
        x = self.conv2(x)
        x = self.gama * x
        x = x.permute(0,3,2,1)  # [N, H, W, C] -> [N, C, H, W]
        x = shortcut+self.drop_path(x)
        return shortcut + x

"""ConvNeXt"""
class ConvNeXt(nn.Module):
    def __init__(self, in_chans: int = 3, num_classes: int = 1000, depths: list = None,
                 dims: list = None, drop_path_rate: float = 0.,
                 layer_scale_init_value: float = 1e-6,
                 head_init_scale: float = 1.):
        super(ConvNeXt, self).__init__()
        self.downsample_layers = nn.ModuleList()
        stem = nn.Sequential(nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
                             LayerNorm(dims[0], eps=1e-6, norm_type="first channel"))
        self.downsample_layers.append(stem)

        # 对应stage2-stage4前的3个downsample
        for i in range(3):
            downsample_layer = nn.Sequential(LayerNorm(dims[i], eps=1e-6, norm_type="first channel"),
                                             nn.Conv2d(dims[i], dims[i+1], kernel_size=2, stride=2))
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList()  # 4 feature resolution stages, each consisting of multiple blocks
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        # 构建每个stage中堆叠的block
        for i in range(4):
            stage = nn.Sequential(
                *[ConvNeXt_block(dim=dims[i],scale_value=layer_scale_init_value)
                  for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.norm = nn.LayerNorm(dims[-1], eps=1e-6)  # final norm layer
        self.head = nn.Linear(dims[-1], num_classes)
        self.apply(self._init_weights)
        self.head.weight.data.mul_(head_init_scale)
        self.head.bias.data.mul_(head_init_scale)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=0.2)
            nn.init.constant_(m.bias, 0)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
        return self.norm(x.mean([-2, -1]))  # global average pooling, (N, C, H, W) -> (N, C)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.forward_features(x)
        x = self.head(x)
        return x