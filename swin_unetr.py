import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np

###############################################################################
# 1) Window Partition & Reverse
###############################################################################
def window_partition(x, window_size):
    """
    Partitions x into non-overlapping windows of size window_size along spatial dims.
    x shape (channels-last, 3D): (batch, D, H, W, C)
    window_size: (d_ws, h_ws, w_ws)

    Returns: (num_windows * batch, window_size.prod(), C)
    """
    b, d, h, w, c = tf.unstack(tf.shape(x))
    d_ws, h_ws, w_ws = window_size

    # Reshape: (b, d//d_ws, d_ws, h//h_ws, h_ws, w//w_ws, w_ws, c)
    x = tf.reshape(x, [
        b,
        d // d_ws, d_ws,
        h // h_ws, h_ws,
        w // w_ws, w_ws,
        c
    ])
    # Transpose: => (b, #D, #H, #W, d_ws, h_ws, w_ws, c)
    x = tf.transpose(x, [0, 1, 3, 5, 2, 4, 6, 7])
    # Flatten: => (b * #windows, d_ws*h_ws*w_ws, c)
    x = tf.reshape(x, [-1, d_ws * h_ws * w_ws, c])
    return x

def window_reverse(windows, window_size, b, d, h, w):
    """
    Reverse window_partition. Takes windows of shape (num_windows * b, window_size.prod(), C)
    and reconstructs a volume of shape (b, d, h, w, C).

    windows shape: (num_windows * b, window_size.prod(), C)
    window_size: (d_ws, h_ws, w_ws)
    """
    d_ws, h_ws, w_ws = window_size
    d_win = d // d_ws
    h_win = h // h_ws
    w_win = w // w_ws

    x = tf.reshape(windows, [b, d_win, h_win, w_win, d_ws, h_ws, w_ws, -1])
    x = tf.transpose(x, [0, 1, 4, 2, 5, 3, 6, 7])  # => (b, d_win, d_ws, h_win, h_ws, w_win, w_ws, C)
    x = tf.reshape(x, [b, d, h, w, -1])
    return x


###############################################################################
# 2) WindowAttention
###############################################################################
class WindowAttention(tf.keras.layers.Layer):
    """
    Window-based multi-head self-attention with relative position bias.
    """

    def __init__(self, dim, window_size, num_heads,
                 qkv_bias=True, attn_drop=0.0, proj_drop=0.0, name="WindowAttention", **kwargs):
        super().__init__(name=name, **kwargs)
        self.dim = dim
        self.window_size = window_size  # (d_ws, h_ws, w_ws)
        self.num_heads = num_heads
        self.qkv_bias = qkv_bias

        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        # Relative position bias table
        self.relative_position_bias_table = self.add_weight(
            shape=((2 * window_size[0] - 1)
                   * (2 * window_size[1] - 1)
                   * (2 * window_size[2] - 1),
                   num_heads),
            initializer="zeros",
            trainable=True,
            name="relative_position_bias_table",
        )
        # We'll compute relative_position_index in build()

        self.qkv = tf.keras.layers.Dense(3 * dim, use_bias=qkv_bias)
        self.attn_drop = tf.keras.layers.Dropout(attn_drop)
        self.proj = tf.keras.layers.Dense(dim)
        self.proj_drop = tf.keras.layers.Dropout(proj_drop)

    def build(self, input_shape):
        super().build(input_shape)
        d_ws, h_ws, w_ws = self.window_size
        coords_d = tf.range(d_ws)
        coords_h = tf.range(h_ws)
        coords_w = tf.range(w_ws)
        coords = tf.stack(tf.meshgrid(coords_d, coords_h, coords_w, indexing="ij"))  # (3, d_ws, h_ws, w_ws)
        coords_flatten = tf.reshape(coords, [3, -1])  # (3, d_ws*h_ws*w_ws)

        # shape => (nTokens, nTokens, 3)
        nTokens = d_ws * h_ws * w_ws
        coords_r = (tf.expand_dims(coords_flatten, -1) - tf.expand_dims(coords_flatten, 1))  # (3, nTokens, nTokens)
        coords_r = tf.transpose(coords_r, [1, 2, 0])  # => (nTokens, nTokens, 3)

        coords_r_d = coords_r[:, :, 0] + (d_ws - 1)
        coords_r_h = coords_r[:, :, 1] + (h_ws - 1)
        coords_r_w = coords_r[:, :, 2] + (w_ws - 1)

        self.relative_position_index = (
            coords_r_d * ((2 * h_ws - 1) * (2 * w_ws - 1))
            + coords_r_h * (2 * w_ws - 1)
            + coords_r_w
        )
        self.relative_position_index = tf.cast(self.relative_position_index, tf.int32)

    def call(self, x, mask=None):
        """
        x: (batch*n_windows, n_tokens, dim)
        """
        b_, n, c = tf.unstack(tf.shape(x))
        qkv = self.qkv(x)  # => (b_, n, 3*dim)
        qkv = tf.reshape(qkv, [b_, n, 3, self.num_heads, c // self.num_heads])
        qkv = tf.transpose(qkv, [2, 0, 3, 1, 4])  # => (3, b_, num_heads, n, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = tf.matmul(q, k, transpose_b=True)  # => (b_, num_heads, n, n)

        # Relative position bias
        nTokens = n
        idx_flat = tf.reshape(self.relative_position_index[:nTokens, :nTokens], [-1])
        relative_bias = tf.gather(self.relative_position_bias_table, idx_flat, axis=0)
        relative_bias = tf.reshape(relative_bias, [nTokens, nTokens, self.num_heads])
        relative_bias = tf.transpose(relative_bias, [2, 0, 1])  # => (num_heads, n, n)
        attn = attn + tf.expand_dims(relative_bias, 0)

        if mask is not None:
            # In practice, you’d handle or expand the mask for all windows
            attn = attn + mask
        attn = tf.nn.softmax(attn, axis=-1)
        attn = self.attn_drop(attn)

        x_out = tf.matmul(attn, v)  # => (b_, num_heads, n, head_dim)
        x_out = tf.transpose(x_out, [0, 2, 1, 3])  # => (b_, n, num_heads, head_dim)
        x_out = tf.reshape(x_out, [b_, n, c])     # => (b_, n, dim)

        x_out = self.proj(x_out)
        x_out = self.proj_drop(x_out)
        return x_out


###############################################################################
# 3) DropPath (Stochastic Depth)
###############################################################################
class DropPath(tf.keras.layers.Layer):
    """
    Stochastic Depth per sample — drops entire path with probability 'drop_prob'.
    """
    def __init__(self, drop_prob=0.0, scale_by_keep=True, **kwargs):
        super().__init__(**kwargs)
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def call(self, x, training=None):
        if (not training) or (self.drop_prob == 0.0):
            return x

        keep_prob = 1.0 - self.drop_prob
        shape = tf.concat([[tf.shape(x)[0]], tf.ones([tf.rank(x) - 1], dtype=tf.int32)], axis=0)
        random_tensor = keep_prob + tf.random.uniform(shape, dtype=x.dtype)
        binary_tensor = tf.floor(random_tensor)

        if self.scale_by_keep:
            x = x / keep_prob
        return x * binary_tensor


###############################################################################
# 4) SwinTransformerBlock & MLP
###############################################################################
class MlpBlock(tf.keras.layers.Layer):
    """
    MLP block used inside the Swin Transformer block.
    """
    def __init__(self, in_features, hidden_features=None, drop=0.0, activation="gelu", **kwargs):
        super().__init__(**kwargs)
        self.in_features = in_features
        self.hidden_features = hidden_features or in_features
        self.dense1 = tf.keras.layers.Dense(self.hidden_features)
        self.act = tf.keras.layers.Activation(activation)
        self.drop1 = tf.keras.layers.Dropout(drop)
        self.dense2 = tf.keras.layers.Dense(in_features)
        self.drop2 = tf.keras.layers.Dropout(drop)

    def call(self, x):
        x = self.dense1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.dense2(x)
        x = self.drop2(x)
        return x

class SwinTransformerBlock(tf.keras.layers.Layer):
    """
    Single Swin Transformer block:
      - window attention (shifted or not), residual
      - MLP, residual
    """
    def __init__(
        self,
        dim,
        num_heads,
        window_size,
        shift_size=(0, 0, 0),
        mlp_ratio=4.0,
        qkv_bias=True,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        name="SwinTransformerBlock",
        **kwargs
    ):
        super().__init__(name=name, **kwargs)
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size

        self.norm1 = tf.keras.layers.LayerNormalization(axis=-1, epsilon=1e-5)
        self.attn = WindowAttention(
            dim=dim,
            window_size=window_size,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.drop_path = DropPath(drop_prob=drop_path)
        self.norm2 = tf.keras.layers.LayerNormalization(axis=-1, epsilon=1e-5)

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MlpBlock(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)

    def call(self, x, mask=None, training=None):
        """
        x shape: (batch, D, H, W, C) channels-last
        """
        input_x = x
        b, d, h, w, c = tf.unstack(tf.shape(x))

        x = self.norm1(x)

        # Shift if needed
        if any(s > 0 for s in self.shift_size):
            x = tf.roll(x, shift=[-self.shift_size[0], -self.shift_size[1], -self.shift_size[2]], axis=[1, 2, 3])

        # Window partition -> attention
        x_windows = window_partition(x, self.window_size)
        attn_windows = self.attn(x_windows, mask)
        x = window_reverse(attn_windows, self.window_size, b, d, h, w)

        # Reverse shift
        if any(s > 0 for s in self.shift_size):
            x = tf.roll(x, shift=[self.shift_size[0], self.shift_size[1], self.shift_size[2]], axis=[1, 2, 3])

        # Residual 1
        x = input_x + self.drop_path(x, training=training)

        # MLP
        x2 = self.norm2(x)
        x2 = self.mlp(x2)
        x = x + self.drop_path(x2, training=training)
        return x


###############################################################################
# 5) PatchMerging & PatchMergingV2
###############################################################################
class PatchMerging(tf.keras.layers.Layer):
    """
    3D patch merging: combine 2x2x2 patches in channels-last format.
    For each 2x2x2 block, we concatenate along channel dimension -> linear transform.
    """
    def __init__(self, dim, name="PatchMerging", **kwargs):
        super().__init__(name=name, **kwargs)
        self.dim = dim

    def build(self, input_shape):
        self.reduction = self.add_weight(
            shape=(8 * self.dim, 2 * self.dim),
            initializer="glorot_uniform",
            trainable=True,
            name="reduction_weight",
        )
        self.bias = self.add_weight(
            shape=(2 * self.dim,),
            initializer="zeros",
            trainable=True,
            name="reduction_bias",
        )
        super().build(input_shape)

    def call(self, x):
        b, d, h, w, c = tf.unstack(tf.shape(x))
        # gather 8 sub-voxels
        x0 = x[:, 0::2, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, 0::2, :]
        x3 = x[:, 0::2, 0::2, 1::2, :]
        x4 = x[:, 1::2, 1::2, 0::2, :]
        x5 = x[:, 1::2, 0::2, 1::2, :]
        x6 = x[:, 0::2, 1::2, 1::2, :]
        x7 = x[:, 1::2, 1::2, 1::2, :]

        x_merged = tf.concat([x0, x1, x2, x3, x4, x5, x6, x7], axis=-1)  # (b, d/2, h/2, w/2, 8*c)
        x_merged = tf.reshape(x_merged, [-1, 8 * self.dim])
        x_merged = tf.matmul(x_merged, self.reduction) + self.bias
        x_merged = tf.reshape(x_merged, [b, d // 2, h // 2, w // 2, 2 * self.dim])
        return x_merged

class PatchMergingV2(tf.keras.layers.Layer):
    """
    Alternate patch merging that applies LayerNorm before the linear reduction.
    """
    def __init__(self, dim, name="PatchMergingV2", **kwargs):
        super().__init__(name=name, **kwargs)
        self.dim = dim

    def build(self, input_shape):
        self.norm = tf.keras.layers.LayerNormalization(axis=-1, epsilon=1e-5)
        self.reduction = self.add_weight(
            shape=(8 * self.dim, 2 * self.dim),
            initializer="glorot_uniform",
            trainable=True,
            name="reduction_weight",
        )
        self.bias = self.add_weight(
            shape=(2 * self.dim,),
            initializer="zeros",
            trainable=True,
            name="reduction_bias",
        )
        super().build(input_shape)

    def call(self, x):
        b, d, h, w, c = tf.unstack(tf.shape(x))

        x0 = x[:, 0::2, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, 0::2, :]
        x3 = x[:, 0::2, 0::2, 1::2, :]
        x4 = x[:, 1::2, 1::2, 0::2, :]
        x5 = x[:, 1::2, 0::2, 1::2, :]
        x6 = x[:, 0::2, 1::2, 1::2, :]
        x7 = x[:, 1::2, 1::2, 1::2, :]

        x_merge = tf.concat([x0, x1, x2, x3, x4, x5, x6, x7], axis=-1)  # => (b, d/2, h/2, w/2, 8*c)
        x_merge = self.norm(x_merge)
        x_merge = tf.reshape(x_merge, [-1, 8 * self.dim])
        x_merge = tf.matmul(x_merge, self.reduction) + self.bias
        x_merge = tf.reshape(x_merge, [b, d // 2, h // 2, w // 2, 2 * self.dim])
        return x_merge


###############################################################################
# 6) BasicLayer
###############################################################################
def linear_spaced(start, end, steps):
    return tf.linspace(start, end, steps)

class BasicLayer(tf.keras.layers.Layer):
    """
    One stage of the Swin Transformer:
      - multiple SwinTransformerBlocks (some with shift)
      - optional downsampling at the end
    """
    def __init__(
        self,
        dim,
        depth,
        num_heads,
        window_size,
        drop_path_rates,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop=0.0,
        attn_drop=0.0,
        downsample=None,
        name="BasicLayer",
        **kwargs
    ):
        super().__init__(name=name, **kwargs)
        self.dim = dim
        self.depth = depth
        self.window_size = window_size
        # We'll alternate shift sizes: (0,0,0) and half-window
        self.shift_sizes = [
            (0, 0, 0),
            tuple(ws // 2 for ws in window_size)
        ]
        self.blocks = []
        for i in range(depth):
            block = SwinTransformerBlock(
                dim=dim,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=self.shift_sizes[i % 2],
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path_rates[i] if isinstance(drop_path_rates, list) else drop_path_rates,
                name=f"{name}_block_{i}",
            )
            self.blocks.append(block)

        self.downsample = downsample(dim=dim) if downsample is not None else None

    def call(self, x, training=None):
        for block in self.blocks:
            x = block(x, training=training)
        if self.downsample is not None:
            x = self.downsample(x)
        return x


###############################################################################
# 7) PatchEmbed
###############################################################################
class PatchEmbed(tf.keras.layers.Layer):
    """
    Splits the input volume into non-overlapping patches and projects to embed_dim.
    """
    def __init__(self, patch_size=(2,2,2), in_chans=1, embed_dim=96, spatial_dims=3,
                 norm_layer=None, name="PatchEmbed", **kwargs):
        super().__init__(name=name, **kwargs)
        self.patch_size = patch_size
        self.spatial_dims = spatial_dims
        self.in_chans = in_chans
        self.embed_dim = embed_dim

        if spatial_dims == 3:
            self.proj = tf.keras.layers.Conv3D(
                filters=embed_dim,
                kernel_size=patch_size,
                strides=patch_size,
                padding="valid"
            )
        else:
            self.proj = tf.keras.layers.Conv2D(
                filters=embed_dim,
                kernel_size=patch_size,
                strides=patch_size,
                padding="valid"
            )

        self.norm = norm_layer(embed_dim) if norm_layer is not None else None

    def call(self, x):
        x = self.proj(x)
        if self.norm:
            x = self.norm(x)
        return x


###############################################################################
# 8) SwinTransformer Backbone
###############################################################################
class SwinTransformer(tf.keras.layers.Layer):
    """
    The Swin Transformer backbone used in SwinUNETR.
    Outputs a list of feature maps at multiple scales.
    """
    def __init__(
        self,
        in_chans,
        embed_dim,
        window_size=(7,7,7),
        patch_size=(2,2,2),
        depths=(2,2,2,2),
        num_heads=(3,6,12,24),
        mlp_ratio=4.0,
        qkv_bias=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        spatial_dims=3,
        downsample="merging",
        name="SwinTransformer",
        **kwargs
    ):
        super().__init__(name=name, **kwargs)
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.window_size = window_size
        self.spatial_dims = spatial_dims

        # patch embed
        self.patch_embed = PatchEmbed(
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            spatial_dims=spatial_dims,
            norm_layer=None
        )

        self.pos_drop = tf.keras.layers.Dropout(drop_rate)
        dpr_vals = linear_spaced(0.0, drop_path_rate, sum(depths)).numpy().tolist()

        self.stages = []
        for i in range(self.num_layers):
            stage_dim = int(embed_dim * (2 ** i))
            if i < self.num_layers - 1:
                # pick merging
                if downsample == "merging":
                    downsample_block = PatchMerging
                else:
                    downsample_block = PatchMergingV2
            else:
                downsample_block = None

            layer = BasicLayer(
                dim=stage_dim,
                depth=depths[i],
                num_heads=num_heads[i],
                window_size=window_size,
                drop_path_rates=dpr_vals[ sum(depths[:i]) : sum(depths[:i+1]) ],
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                downsample=downsample_block,
                name=f"swin_stage_{i}"
            )
            self.stages.append(layer)

        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))

    def call(self, x, normalize_out=True, training=None):
        # patch embed
        x = self.patch_embed(x)
        x = self.pos_drop(x, training=training)

        outputs = []
        # First output is the patch-embedding result
        x0_out = x
        outputs.append(self._norm_if(x0_out, normalize_out))

        # Go through each stage
        cur = x
        for i, stage in enumerate(self.stages):
            cur = stage(cur, training=training)
            outputs.append(self._norm_if(cur, normalize_out))

        return outputs

    def _norm_if(self, x, normalize_out):
        if not normalize_out:
            return x
        ln = tf.keras.layers.LayerNormalization(axis=-1, epsilon=1e-5)
        return ln(x)


###############################################################################
# 9) UNet Decoder Blocks
###############################################################################
def get_norm_layer_1d(norm_name, out_channels):
    if norm_name == "batch":
        return tf.keras.layers.BatchNormalization(axis=-1)
    elif norm_name == "layer":
        return tf.keras.layers.LayerNormalization(axis=-1, epsilon=1e-5)
    else:  # default "instance"
        return tfa.layers.InstanceNormalization(axis=-1)

class UnetrBasicBlock(tf.keras.Model):
    """
    [Conv -> Norm -> ReLU]*2, optional residual
    """
    def __init__(
        self,
        spatial_dims,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        norm_name="instance",
        res_block=True,
        name="UnetrBasicBlock",
        **kwargs
    ):
        super().__init__(name=name, **kwargs)
        self.res_block = res_block
        if spatial_dims == 3:
            conv_class = tf.keras.layers.Conv3D
        else:
            conv_class = tf.keras.layers.Conv2D

        self.conv1 = conv_class(out_channels, kernel_size=kernel_size, strides=stride, padding='same')
        self.norm1 = get_norm_layer_1d(norm_name, out_channels)
        self.act1 = tf.keras.layers.ReLU()

        self.conv2 = conv_class(out_channels, kernel_size=kernel_size, strides=stride, padding='same')
        self.norm2 = get_norm_layer_1d(norm_name, out_channels)
        self.act2 = tf.keras.layers.ReLU()

        self.shortcut = None
        if self.res_block and (in_channels != out_channels):
            self.shortcut = conv_class(out_channels, kernel_size=1, strides=1, padding='same')

    def call(self, x):
        residual = x
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.act1(x)

        x = self.conv2(x)
        x = self.norm2(x)

        if self.res_block:
            if self.shortcut is not None:
                residual = self.shortcut(residual)
            x = x + residual
        x = self.act2(x)
        return x

class UnetrUpBlock(tf.keras.Model):
    """
    Upsample -> concat skip -> UnetrBasicBlock
    """
    def __init__(
        self,
        spatial_dims,
        in_channels,
        out_channels,
        kernel_size=3,
        upsample_kernel_size=2,
        norm_name="instance",
        res_block=True,
        name="UnetrUpBlock",
        **kwargs
    ):
        super().__init__(name=name, **kwargs)
        if spatial_dims == 3:
            self.upsample = tf.keras.layers.UpSampling3D(size=upsample_kernel_size)
        else:
            self.upsample = tf.keras.layers.UpSampling2D(size=upsample_kernel_size)

        self.conv_block = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=in_channels + out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=1,
            norm_name=norm_name,
            res_block=res_block
        )

    def call(self, x, skip):
        x = self.upsample(x)
        x = tf.concat([x, skip], axis=-1)
        x = self.conv_block(x)
        return x

class UnetOutBlock(tf.keras.Model):
    """
    Final 1x1 conv for desired out_channels.
    """
    def __init__(self, spatial_dims, in_channels, out_channels, name="UnetOutBlock", **kwargs):
        super().__init__(name=name, **kwargs)
        if spatial_dims == 3:
            self.conv = tf.keras.layers.Conv3D(out_channels, kernel_size=1, padding="same")
        else:
            self.conv = tf.keras.layers.Conv2D(out_channels, kernel_size=1, padding="same")

    def call(self, x):
        return self.conv(x)


###############################################################################
# 10) The SwinUNETR Model
###############################################################################
class SwinUNETR(tf.keras.Model):
    """
    TensorFlow 2 Implementation of Swin-UNETR, mirroring MONAI's approach.
    """

    def __init__(
        self,
        img_size=(96, 96, 96),
        in_channels=1,
        out_channels=2,
        depths=(2, 2, 2, 2),
        num_heads=(3, 6, 12, 24),
        feature_size=24,
        norm_name="instance",
        drop_rate=0.0,
        attn_drop_rate=0.0,
        dropout_path_rate=0.0,
        spatial_dims=3,
        downsample="merging",
        name="SwinUNETR",
        **kwargs
    ):
        super().__init__(name=name, **kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels

        window_size = (7,7,7) if spatial_dims == 3 else (7,7)
        patch_size = (2,2,2) if spatial_dims == 3 else (2,2)

        self.swinViT = SwinTransformer(
            in_chans=in_channels,
            embed_dim=feature_size,
            window_size=window_size,
            patch_size=patch_size,
            depths=depths,
            num_heads=num_heads,
            mlp_ratio=4.0,
            qkv_bias=True,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=dropout_path_rate,
            spatial_dims=spatial_dims,
            downsample=downsample
        )

        self.encoder1 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
            name="encoder1"
        )

        self.encoder2 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size,
            out_channels=feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
            name="encoder2"
        )

        self.encoder3 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=2*feature_size,
            out_channels=2*feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
            name="encoder3"
        )

        self.encoder4 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=4*feature_size,
            out_channels=4*feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
            name="encoder4"
        )

        self.encoder10 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=16*feature_size,
            out_channels=16*feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
            name="encoder10"
        )

        self.decoder5 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=16*feature_size,
            out_channels=8*feature_size,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
            name="decoder5"
        )
        self.decoder4 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=8*feature_size,
            out_channels=4*feature_size,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
            name="decoder4"
        )
        self.decoder3 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=4*feature_size,
            out_channels=2*feature_size,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
            name="decoder3"
        )
        self.decoder2 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=2*feature_size,
            out_channels=feature_size,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
            name="decoder2"
        )
        self.decoder1 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size,
            out_channels=feature_size,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
            name="decoder1"
        )

        self.out = UnetOutBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size,
            out_channels=out_channels,
            name="out"
        )

    def call(self, x, training=None):
        """
        Forward pass:
        hidden_states = self.swinViT(x) => [x0, x1, x2, x3, x4]
        """
        enc0 = self.encoder1(x)  # raw input => feature_size

        hidden_states = self.swinViT(x, training=training, normalize_out=True)
        # hidden_states[0] => patch embed
        # hidden_states[1] => stage1
        # hidden_states[2] => stage2
        # hidden_states[3] => stage3
        # hidden_states[4] => stage4

        enc1 = self.encoder2(hidden_states[0])
        enc2 = self.encoder3(hidden_states[1])
        enc3 = self.encoder4(hidden_states[2])
        dec4 = self.encoder10(hidden_states[4])

        dec3 = self.decoder5(dec4, hidden_states[3])
        dec2 = self.decoder4(dec3, enc3)
        dec1 = self.decoder3(dec2, enc2)
        dec0 = self.decoder2(dec1, enc1)
        out = self.decoder1(dec0, enc0)

        logits = self.out(out)
        return logits


###############################################################################
# 11) Example Usage
###############################################################################
if __name__ == "__main__":
    # Example usage
    model = SwinUNETR(
        img_size=(56, 56, 56),
        in_channels=1,
        out_channels=2,
        feature_size=24,
        depths=(2,2,2,2),
        num_heads=(3,6,12,24),
        spatial_dims=3
    )

    dummy_input = tf.random.normal([1, 56, 56, 56, 1])
    output = model(dummy_input)
    print("Output shape:", output.shape)  # e.g. (1, 96, 96, 96, 2)

    """
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )
    """
    model.summary()