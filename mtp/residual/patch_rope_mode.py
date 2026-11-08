p = "/u701/j692wu/swift-opt/mtp/residual/mtp_torch.py"
s = open(p).read()
old = """    freqs = positions.to(torch.float32)[..., None] * inv_freq  # [B, T, half]
    cos = freqs.cos().to(x.dtype)[:, :, None, :]
    sin = freqs.sin().to(x.dtype)[:, :, None, :]
    xr, xp = x[..., :rotary_dim], x[..., rotary_dim:]
    x1, x2 = xr[..., :half], xr[..., half:]
    o1 = x1 * cos - x2 * sin
    o2 = x2 * cos + x1 * sin
    return torch.cat([o1, o2, xp], dim=-1)
"""
new = """    pos = positions.to(torch.float32)
    if ROPE_MODE == "shift+1":
        pos = pos + 1.0
    elif ROPE_MODE == "shift-1":
        pos = pos - 1.0
    freqs = pos[..., None] * inv_freq  # [B, T, half]
    if ROPE_MODE == "fp32cs":
        cos = freqs.cos()[:, :, None, :]
        sin = freqs.sin()[:, :, None, :]
        xf = x.float()
        xr, xp = xf[..., :rotary_dim], xf[..., rotary_dim:]
        x1, x2 = xr[..., :half], xr[..., half:]
        o1 = x1 * cos - x2 * sin
        o2 = x2 * cos + x1 * sin
        return torch.cat([o1, o2, xp], dim=-1).to(x.dtype)
    cos = freqs.cos().to(x.dtype)[:, :, None, :]
    sin = freqs.sin().to(x.dtype)[:, :, None, :]
    if ROPE_MODE == "fused":
        # ops: fused_qk_norm_rope.py:125-126 -- fp32 math on bf16 cos/sin, single rounding at store
        xf = x.float()
        cf, sf = cos.float(), sin.float()
        xr, xp = xf[..., :rotary_dim], xf[..., rotary_dim:]
        x1, x2 = xr[..., :half], xr[..., half:]
        o1 = x1 * cf - x2 * sf
        o2 = x2 * cf + x1 * sf
        return torch.cat([o1, o2, xp], dim=-1).to(x.dtype)
    xr, xp = x[..., :rotary_dim], x[..., rotary_dim:]
    x1, x2 = xr[..., :half], xr[..., half:]
    o1 = x1 * cos - x2 * sin
    o2 = x2 * cos + x1 * sin
    return torch.cat([o1, o2, xp], dim=-1)
"""
assert s.count(old) == 1, s.count(old)
s = s.replace(old, new)
anchor = "ATTN_MODE = "
i = s.index(anchor); j = s.index(chr(10), i) + 1
s = s[:j] + "ROPE_MODE = \"bf16\"  # residual experiments: bf16 | fused | fp32cs | shift+1 | shift-1\n" + s[j:]
open(p, "w").write(s)
print("patched")
