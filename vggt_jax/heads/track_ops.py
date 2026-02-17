import jax
import jax.numpy as jnp

from vggt_jax.holders import AttnBlockHolder, CrossAttnBlockHolder, TorchMlpHolder, TorchMultiHeadAttentionHolder, TrackerHolder


def _torch_linear(holder, x: jnp.ndarray) -> jnp.ndarray:
    y = jnp.matmul(x, holder.weight[...].T, precision=jax.lax.Precision.DEFAULT)
    if hasattr(holder, "bias"):
        y = y + holder.bias[...]
    return y


def _layer_norm(x: jnp.ndarray, holder, eps: float = 1e-5) -> jnp.ndarray:
    mean = jnp.mean(x, axis=-1, keepdims=True)
    var = jnp.mean((x - mean) ** 2, axis=-1, keepdims=True)
    x_hat = (x - mean) / jnp.sqrt(var + eps)
    return x_hat * holder.weight[...] + holder.bias[...]


def _group_norm_g1(x: jnp.ndarray, holder, eps: float = 1e-5) -> jnp.ndarray:
    mean = jnp.mean(x, axis=-1, keepdims=True)
    var = jnp.mean((x - mean) ** 2, axis=-1, keepdims=True)
    x_hat = (x - mean) / jnp.sqrt(var + eps)
    return x_hat * holder.weight[...] + holder.bias[...]


def _torch_mlp(x: jnp.ndarray, holder: TorchMlpHolder) -> jnp.ndarray:
    x = _torch_linear(holder.fc1, x)
    x = jax.nn.gelu(x, approximate=False)
    x = _torch_linear(holder.fc2, x)
    return x


def _split_heads(x: jnp.ndarray, num_heads: int) -> jnp.ndarray:
    batch_size, tokens, channels = x.shape
    head_dim = channels // num_heads
    return x.reshape(batch_size, tokens, num_heads, head_dim).transpose(0, 2, 1, 3)


def _merge_heads(x: jnp.ndarray) -> jnp.ndarray:
    batch_size, num_heads, tokens, head_dim = x.shape
    return x.transpose(0, 2, 1, 3).reshape(batch_size, tokens, num_heads * head_dim)


def _mha_self(x: jnp.ndarray, holder: TorchMultiHeadAttentionHolder, num_heads: int) -> jnp.ndarray:
    in_proj_weight = holder.in_proj_weight[...]
    in_proj_bias = holder.in_proj_bias[...]

    batch_size, tokens, channels = x.shape
    head_dim = channels // num_heads

    qkv = jnp.matmul(x, in_proj_weight.T, precision=jax.lax.Precision.DEFAULT) + in_proj_bias
    q, k, v = jnp.split(qkv, 3, axis=-1)

    def reshape_for_attention(tensor: jnp.ndarray) -> jnp.ndarray:
        tensor = tensor.reshape(batch_size, tokens, num_heads, head_dim).transpose(0, 2, 1, 3)
        return tensor.reshape(batch_size * num_heads, tokens, head_dim)

    q = reshape_for_attention(q)
    k = reshape_for_attention(k)
    v = reshape_for_attention(v)

    q = q * (1.0 / jnp.sqrt(jnp.asarray(head_dim, dtype=x.dtype)))
    attn = jnp.matmul(q, jnp.swapaxes(k, 1, 2), precision=jax.lax.Precision.DEFAULT)
    attn = jax.nn.softmax(attn, axis=-1)
    out = jnp.matmul(attn, v, precision=jax.lax.Precision.DEFAULT)

    out = out.reshape(batch_size, num_heads, tokens, head_dim).transpose(0, 2, 1, 3).reshape(batch_size, tokens, channels)
    return _torch_linear(holder.out_proj, out)


def _mha_cross(
    x: jnp.ndarray,
    context: jnp.ndarray,
    holder: TorchMultiHeadAttentionHolder,
    num_heads: int,
) -> jnp.ndarray:
    in_proj_weight = holder.in_proj_weight[...]
    in_proj_bias = holder.in_proj_bias[...]

    hidden = x.shape[-1]
    w_q, w_k, w_v = jnp.split(in_proj_weight, 3, axis=0)
    b_q, b_k, b_v = jnp.split(in_proj_bias, 3, axis=0)

    batch_size, tokens_q, _ = x.shape
    tokens_k = context.shape[1]
    head_dim = hidden // num_heads

    q = jnp.matmul(x, w_q.T, precision=jax.lax.Precision.DEFAULT) + b_q
    k = jnp.matmul(context, w_k.T, precision=jax.lax.Precision.DEFAULT) + b_k
    v = jnp.matmul(context, w_v.T, precision=jax.lax.Precision.DEFAULT) + b_v

    def reshape_qkv(tensor: jnp.ndarray, tokens: int) -> jnp.ndarray:
        tensor = tensor.reshape(batch_size, tokens, num_heads, head_dim).transpose(0, 2, 1, 3)
        return tensor.reshape(batch_size * num_heads, tokens, head_dim)

    q = reshape_qkv(q, tokens_q)
    k = reshape_qkv(k, tokens_k)
    v = reshape_qkv(v, tokens_k)

    q = q * (1.0 / jnp.sqrt(jnp.asarray(head_dim, dtype=x.dtype)))
    attn = jnp.matmul(q, jnp.swapaxes(k, 1, 2), precision=jax.lax.Precision.DEFAULT)
    attn = jax.nn.softmax(attn, axis=-1)
    out = jnp.matmul(attn, v, precision=jax.lax.Precision.DEFAULT)

    out = out.reshape(batch_size, num_heads, tokens_q, head_dim).transpose(0, 2, 1, 3).reshape(batch_size, tokens_q, hidden)
    return _torch_linear(holder.out_proj, out)


def _attn_block(x: jnp.ndarray, holder: AttnBlockHolder, num_heads: int) -> jnp.ndarray:
    x = _layer_norm(x, holder.norm1)
    x = x + _mha_self(x, holder.attn, num_heads)
    x = x + _torch_mlp(_layer_norm(x, holder.norm2), holder.mlp)
    return x


def _cross_attn_block(
    x: jnp.ndarray,
    context: jnp.ndarray,
    holder: CrossAttnBlockHolder,
    num_heads: int,
) -> jnp.ndarray:
    x = _layer_norm(x, holder.norm1)
    context = _layer_norm(context, holder.norm_context)
    x = x + _mha_cross(x, context, holder.cross_attn, num_heads)
    x = x + _torch_mlp(_layer_norm(x, holder.norm2), holder.mlp)
    return x


def _efficient_updateformer(
    x: jnp.ndarray,
    updateformer,
    *,
    num_heads: int,
) -> jnp.ndarray:
    tokens = _layer_norm(x, updateformer.input_norm)
    tokens = _torch_linear(updateformer.input_transform, tokens)
    init_tokens = tokens

    batch_size, num_points, seq_len, hidden_size = tokens.shape

    num_virtual_tracks = 0
    if hasattr(updateformer, "virual_tracks"):
        num_virtual_tracks = updateformer.virual_tracks.shape[1]
        virtual_tokens = jnp.broadcast_to(
            updateformer.virual_tracks[...],
            (batch_size, num_virtual_tracks, seq_len, hidden_size),
        )
        tokens = jnp.concatenate([tokens, virtual_tokens], axis=1)

    total_tracks = tokens.shape[1]

    num_time_blocks = len(updateformer.time_blocks)
    has_space_attn = hasattr(updateformer, "space_virtual_blocks") and len(updateformer.space_virtual_blocks) > 0
    space_interval = None
    if has_space_attn:
        num_space_blocks = len(updateformer.space_virtual_blocks)
        space_interval = max(num_time_blocks // max(num_space_blocks, 1), 1)

    space_index = 0
    for time_index in range(num_time_blocks):
        time_tokens = tokens.reshape(batch_size * total_tracks, seq_len, hidden_size)
        time_tokens = _attn_block(time_tokens, updateformer.time_blocks[time_index], num_heads)
        tokens = time_tokens.reshape(batch_size, total_tracks, seq_len, hidden_size)

        if has_space_attn and space_interval is not None and (time_index % space_interval == 0):
            if space_index >= len(updateformer.space_virtual_blocks):
                continue

            space_tokens = tokens.transpose(0, 2, 1, 3).reshape(batch_size * seq_len, total_tracks, hidden_size)

            point_tokens = space_tokens[:, : total_tracks - num_virtual_tracks]
            virtual_tokens = space_tokens[:, total_tracks - num_virtual_tracks :]

            virtual_tokens = _cross_attn_block(
                virtual_tokens,
                point_tokens,
                updateformer.space_virtual2point_blocks[space_index],
                num_heads,
            )
            virtual_tokens = _attn_block(virtual_tokens, updateformer.space_virtual_blocks[space_index], num_heads)
            point_tokens = _cross_attn_block(
                point_tokens,
                virtual_tokens,
                updateformer.space_point2virtual_blocks[space_index],
                num_heads,
            )

            space_tokens = jnp.concatenate([point_tokens, virtual_tokens], axis=1)
            tokens = space_tokens.reshape(batch_size, seq_len, total_tracks, hidden_size).transpose(0, 2, 1, 3)
            space_index += 1

    if num_virtual_tracks > 0:
        tokens = tokens[:, : total_tracks - num_virtual_tracks]

    tokens = tokens + init_tokens
    tokens = _layer_norm(tokens, updateformer.output_norm)
    flow = _torch_linear(updateformer.flow_head, tokens)
    return flow


def _avg_pool2d_nchw(x: jnp.ndarray, stride: int) -> jnp.ndarray:
    if stride <= 1:
        return x
    pooled = jax.lax.reduce_window(
        x,
        0.0,
        jax.lax.add,
        window_dimensions=(1, 1, stride, stride),
        window_strides=(1, 1, stride, stride),
        padding="VALID",
    )
    return pooled / float(stride * stride)


def avg_pool2d_nhwc(x: jnp.ndarray, stride: int) -> jnp.ndarray:
    if stride <= 1:
        return x
    pooled = jax.lax.reduce_window(
        x,
        0.0,
        jax.lax.add,
        window_dimensions=(1, stride, stride, 1),
        window_strides=(1, stride, stride, 1),
        padding="VALID",
    )
    return pooled / float(stride * stride)


def _sample_nchw(input_tensor: jnp.ndarray, coords: jnp.ndarray, padding_mode: str = "border") -> jnp.ndarray:
    batch_size, channels, height, width = input_tensor.shape
    coords = coords.astype(input_tensor.dtype)

    out_shape = coords.shape[1:-1]
    coords_flat = coords.reshape(batch_size, -1, 2)

    # Match upstream bilinear_sampler -> grid_sample semantics:
    # bilinear_sampler scales pixel coordinates into [-1, 1] and grid_sample
    # unnormalizes them back into pixel space. Replicating that round-trip here
    # avoids tiny float32 discrepancies that get amplified across iterations.
    dtype = input_tensor.dtype

    scale_x = jnp.asarray(2.0, dtype=dtype) / jnp.asarray(max(width - 1, 1), dtype=dtype)
    scale_y = jnp.asarray(2.0, dtype=dtype) / jnp.asarray(max(height - 1, 1), dtype=dtype)
    x_norm = coords_flat[..., 0] * scale_x - jnp.asarray(1.0, dtype=dtype)
    y_norm = coords_flat[..., 1] * scale_y - jnp.asarray(1.0, dtype=dtype)

    x = (x_norm + jnp.asarray(1.0, dtype=dtype)) * jnp.asarray(width - 1, dtype=dtype) / jnp.asarray(2.0, dtype=dtype)
    y = (y_norm + jnp.asarray(1.0, dtype=dtype)) * jnp.asarray(height - 1, dtype=dtype) / jnp.asarray(2.0, dtype=dtype)

    if width == 1:
        x = jnp.zeros_like(x)
    if height == 1:
        y = jnp.zeros_like(y)

    if padding_mode == "border":
        x = jnp.clip(x, 0.0, float(max(width - 1, 0)))
        y = jnp.clip(y, 0.0, float(max(height - 1, 0)))

    x0 = jnp.floor(x).astype(jnp.int32)
    y0 = jnp.floor(y).astype(jnp.int32)
    x1 = x0 + 1
    y1 = y0 + 1

    x0_clip = jnp.clip(x0, 0, max(width - 1, 0))
    y0_clip = jnp.clip(y0, 0, max(height - 1, 0))
    x1_clip = jnp.clip(x1, 0, max(width - 1, 0))
    y1_clip = jnp.clip(y1, 0, max(height - 1, 0))

    batch_idx = jnp.arange(batch_size, dtype=jnp.int32)[:, None]
    ia = input_tensor[batch_idx, :, y0_clip, x0_clip]
    ib = input_tensor[batch_idx, :, y1_clip, x0_clip]
    ic = input_tensor[batch_idx, :, y0_clip, x1_clip]
    id_ = input_tensor[batch_idx, :, y1_clip, x1_clip]

    x0_f = x0.astype(input_tensor.dtype)
    y0_f = y0.astype(input_tensor.dtype)
    x1_f = x1.astype(input_tensor.dtype)
    y1_f = y1.astype(input_tensor.dtype)

    wa = (x1_f - x) * (y1_f - y)
    wb = (x1_f - x) * (y - y0_f)
    wc = (x - x0_f) * (y1_f - y)
    wd = (x - x0_f) * (y - y0_f)

    if padding_mode == "zeros":
        valid_a = ((x0 >= 0) & (x0 < width) & (y0 >= 0) & (y0 < height)).astype(input_tensor.dtype)
        valid_b = ((x0 >= 0) & (x0 < width) & (y1 >= 0) & (y1 < height)).astype(input_tensor.dtype)
        valid_c = ((x1 >= 0) & (x1 < width) & (y0 >= 0) & (y0 < height)).astype(input_tensor.dtype)
        valid_d = ((x1 >= 0) & (x1 < width) & (y1 >= 0) & (y1 < height)).astype(input_tensor.dtype)
        ia = ia * valid_a[..., None]
        ib = ib * valid_b[..., None]
        ic = ic * valid_c[..., None]
        id_ = id_ * valid_d[..., None]

    output = (
        wa[..., None] * ia
        + wb[..., None] * ib
        + wc[..., None] * ic
        + wd[..., None] * id_
    )

    output = output.reshape(batch_size, *out_shape, channels)
    output = output.transpose(0, output.ndim - 1, *range(1, output.ndim - 1))
    return output


def sample_features4d(input_tensor: jnp.ndarray, coords: jnp.ndarray) -> jnp.ndarray:
    sampled = _sample_nchw(input_tensor, coords, padding_mode="border")
    return sampled.transpose(0, 2, 1)


def _compute_corr_level(fmap1: jnp.ndarray, fmap2s: jnp.ndarray, channels: int) -> jnp.ndarray:
    corrs = jnp.einsum("bsnc,bsch->bsnh", fmap1, fmap2s, precision=jax.lax.Precision.HIGHEST)
    return corrs / jnp.sqrt(jnp.asarray(channels, dtype=fmap1.dtype))


class CorrBlock:
    def __init__(
        self,
        fmaps: jnp.ndarray,
        *,
        num_levels: int = 4,
        radius: int = 4,
        padding_mode: str = "zeros",
    ):
        self.num_levels = num_levels
        self.radius = radius
        self.padding_mode = padding_mode

        self.fmaps_pyramid = [fmaps]
        current_fmaps = fmaps
        for _ in range(num_levels - 1):
            batch_size, seq_len, channels, height, width = current_fmaps.shape
            current_fmaps = current_fmaps.reshape(batch_size * seq_len, channels, height, width)
            current_fmaps = _avg_pool2d_nchw(current_fmaps, 2)
            _, _, h_new, w_new = current_fmaps.shape
            current_fmaps = current_fmaps.reshape(batch_size, seq_len, channels, h_new, w_new)
            self.fmaps_pyramid.append(current_fmaps)

        delta_values = jnp.linspace(-radius, radius, 2 * radius + 1, dtype=fmaps.dtype)
        delta_grid = jnp.meshgrid(delta_values, delta_values, indexing="ij")
        self.delta = jnp.stack(delta_grid, axis=-1)

    def corr_sample(self, targets: jnp.ndarray, coords: jnp.ndarray) -> jnp.ndarray:
        batch_size, seq_len, num_tracks, _ = targets.shape
        out_pyramid = []

        for level, fmaps in enumerate(self.fmaps_pyramid):
            _, _, channels, height, width = fmaps.shape
            fmap2s = fmaps.reshape(batch_size, seq_len, channels, height * width)
            corrs = _compute_corr_level(targets, fmap2s, channels)
            corrs = corrs.reshape(batch_size, seq_len, num_tracks, height, width)

            centroid = coords.reshape(batch_size * seq_len * num_tracks, 1, 1, 2) / float(2**level)
            coords_lvl = centroid + self.delta.reshape(1, 2 * self.radius + 1, 2 * self.radius + 1, 2)

            corrs_sampled = _sample_nchw(
                corrs.reshape(batch_size * seq_len * num_tracks, 1, height, width),
                coords_lvl,
                padding_mode=self.padding_mode,
            )
            corrs_sampled = corrs_sampled.reshape(batch_size, seq_len, num_tracks, -1)
            out_pyramid.append(corrs_sampled)

        return jnp.concatenate(out_pyramid, axis=-1)


def _get_1d_sincos_pos_embed_from_grid(embed_dim: int, pos: jnp.ndarray) -> jnp.ndarray:
    omega = jnp.arange(embed_dim // 2, dtype=jnp.float32)
    omega = omega / (embed_dim / 2.0)
    omega = 1.0 / (10000.0**omega)

    pos = pos.reshape(-1).astype(jnp.float32)
    out = jnp.einsum("m,d->md", pos, omega, precision=jax.lax.Precision.HIGHEST)

    emb_sin = jnp.sin(out)
    emb_cos = jnp.cos(out)
    emb = jnp.concatenate([emb_sin, emb_cos], axis=1)
    return emb[None]


def _get_2d_sincos_pos_embed_from_grid(embed_dim: int, grid: jnp.ndarray) -> jnp.ndarray:
    emb_h = _get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])
    emb_w = _get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])
    return jnp.concatenate([emb_h, emb_w], axis=2)


def get_2d_sincos_pos_embed(embed_dim: int, grid_size: tuple[int, int] | int) -> jnp.ndarray:
    if isinstance(grid_size, tuple):
        grid_h, grid_w = grid_size
    else:
        grid_h = grid_w = grid_size

    grid_h_arr = jnp.arange(grid_h, dtype=jnp.float32)
    grid_w_arr = jnp.arange(grid_w, dtype=jnp.float32)
    grid = jnp.meshgrid(grid_w_arr, grid_h_arr, indexing="xy")
    grid = jnp.stack(grid, axis=0)
    grid = grid.reshape(2, 1, grid_h, grid_w)

    pos_embed = _get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    return pos_embed.reshape(1, grid_h, grid_w, embed_dim).transpose(0, 3, 1, 2)


def get_2d_embedding(xy: jnp.ndarray, channels: int, cat_coords: bool = True) -> jnp.ndarray:
    batch_size, num_points, dims = xy.shape
    if dims != 2:
        raise ValueError(f"Expected xy with last dim 2, got {dims}")

    x = xy[:, :, 0:1]
    y = xy[:, :, 1:2]
    div_term = (jnp.arange(0, channels, 2, dtype=jnp.float32) * (1000.0 / channels)).reshape(1, 1, channels // 2)

    pe_x = jnp.zeros((batch_size, num_points, channels), dtype=jnp.float32)
    pe_y = jnp.zeros((batch_size, num_points, channels), dtype=jnp.float32)

    pe_x = pe_x.at[:, :, 0::2].set(jnp.sin(x * div_term))
    pe_x = pe_x.at[:, :, 1::2].set(jnp.cos(x * div_term))

    pe_y = pe_y.at[:, :, 0::2].set(jnp.sin(y * div_term))
    pe_y = pe_y.at[:, :, 1::2].set(jnp.cos(y * div_term))

    pe = jnp.concatenate([pe_x, pe_y], axis=2)
    if cat_coords:
        pe = jnp.concatenate([xy, pe], axis=2)
    return pe


def run_tracker_predictor(
    tracker: TrackerHolder,
    *,
    query_points: jnp.ndarray,
    fmaps: jnp.ndarray,
    iters: int,
    stride: int,
    corr_levels: int,
    corr_radius: int,
    max_scale: int,
    num_heads: int,
    apply_sigmoid: bool = True,
) -> tuple[list[jnp.ndarray], jnp.ndarray, jnp.ndarray | None]:
    batch_size, num_tracks, dims = query_points.shape
    b_fmap, seq_len, latent_dim, fmap_h, fmap_w = fmaps.shape
    if batch_size != b_fmap:
        raise ValueError(f"query_points batch {batch_size} != fmap batch {b_fmap}")
    if dims != 2:
        raise ValueError(f"query_points last dim must be 2, got {dims}")

    fmaps = fmaps.transpose(0, 1, 3, 4, 2)
    fmaps = _layer_norm(fmaps, tracker.fmap_norm)
    fmaps = fmaps.transpose(0, 1, 4, 2, 3)

    query_points = query_points / float(stride)

    coords = jnp.broadcast_to(query_points[:, None, :, :], (batch_size, seq_len, num_tracks, 2))

    query_track_feat = sample_features4d(fmaps[:, 0], coords[:, 0])
    track_feats = jnp.broadcast_to(query_track_feat[:, None, :, :], (batch_size, seq_len, num_tracks, latent_dim))
    coords_backup = coords

    corr_fn = CorrBlock(
        fmaps,
        num_levels=corr_levels,
        radius=corr_radius,
        padding_mode="zeros",
    )

    coord_preds: list[jnp.ndarray] = []

    flows_emb_dim = latent_dim // 2

    for _ in range(iters):
        coords = jax.lax.stop_gradient(coords)
        fcorrs = corr_fn.corr_sample(track_feats, coords)

        corr_dim = fcorrs.shape[3]
        fcorrs_ = fcorrs.transpose(0, 2, 1, 3).reshape(batch_size * num_tracks, seq_len, corr_dim)
        fcorrs_ = _torch_mlp(fcorrs_, tracker.corr_mlp)

        flows = (coords - coords[:, 0:1]).transpose(0, 2, 1, 3).reshape(
            batch_size * num_tracks, seq_len, 2
        )
        flows_emb = get_2d_embedding(flows, flows_emb_dim, cat_coords=False)
        flows_emb = jnp.concatenate([flows_emb, flows / float(max_scale), flows / float(max_scale)], axis=-1)

        track_feats_ = track_feats.transpose(0, 2, 1, 3).reshape(batch_size * num_tracks, seq_len, latent_dim)
        transformer_input = jnp.concatenate([flows_emb, fcorrs_, track_feats_], axis=2)

        transformer_dim = transformer_input.shape[-1]
        pos_embed = get_2d_sincos_pos_embed(transformer_dim, grid_size=(fmap_h, fmap_w)).astype(query_points.dtype)
        pos_embed = jnp.broadcast_to(pos_embed, (batch_size, *pos_embed.shape[1:]))
        sampled_pos_emb = sample_features4d(pos_embed, coords[:, 0])
        sampled_pos_emb = sampled_pos_emb.reshape(batch_size * num_tracks, 1, transformer_dim)

        x = transformer_input + sampled_pos_emb

        ref_head = tracker.query_ref_token[:, 0:1, :]
        if seq_len > 1:
            ref_tail = jnp.broadcast_to(tracker.query_ref_token[:, 1:2, :], (1, seq_len - 1, transformer_dim))
            query_ref = jnp.concatenate([ref_head, ref_tail], axis=1)
        else:
            query_ref = ref_head
        x = x + query_ref

        x = x.reshape(batch_size, num_tracks, seq_len, transformer_dim)
        delta = _efficient_updateformer(x, tracker.updateformer, num_heads=num_heads)

        delta = delta.reshape(batch_size * num_tracks, seq_len, latent_dim + 2)
        delta_coords = delta[:, :, :2]
        delta_feats = delta[:, :, 2:]

        track_feats_flat = track_feats_.reshape(batch_size * num_tracks * seq_len, latent_dim)
        delta_feats = delta_feats.reshape(batch_size * num_tracks * seq_len, latent_dim)

        delta_feats = _group_norm_g1(delta_feats, tracker.ffeat_norm)
        delta_feats = _torch_linear(tracker.ffeat_updater[0], delta_feats)
        delta_feats = jax.nn.gelu(delta_feats, approximate=False)

        track_feats_flat = delta_feats + track_feats_flat

        track_feats = track_feats_flat.reshape(batch_size, num_tracks, seq_len, latent_dim).transpose(0, 2, 1, 3)

        coords = coords + delta_coords.reshape(batch_size, num_tracks, seq_len, 2).transpose(0, 2, 1, 3)
        coords = coords.at[:, 0].set(coords_backup[:, 0])

        coord_preds.append(coords * float(stride))

    vis = _torch_linear(tracker.vis_predictor[0], track_feats.reshape(batch_size * seq_len * num_tracks, latent_dim))
    vis = vis.reshape(batch_size, seq_len, num_tracks)
    if apply_sigmoid:
        vis = jax.nn.sigmoid(vis)

    conf = None
    if hasattr(tracker, "conf_predictor"):
        conf = _torch_linear(tracker.conf_predictor[0], track_feats.reshape(batch_size * seq_len * num_tracks, latent_dim))
        conf = conf.reshape(batch_size, seq_len, num_tracks)
        if apply_sigmoid:
            conf = jax.nn.sigmoid(conf)

    return coord_preds, vis, conf
