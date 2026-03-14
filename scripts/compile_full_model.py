#!/usr/bin/env python3
"""Full multi-layer stateful transformer decoder — all layers in ONE CoreML model.

Uses exec() to dynamically generate a function with the correct number of
state parameters (coremltools requires explicit named parameters).

Usage:
    python3 scripts/compile_full_model.py --layers 4 --D 896
    python3 scripts/compile_full_model.py --layers 24 --D 896 --max-seq-len 512
"""

import argparse, os, textwrap, numpy as np
import coremltools as ct
from coremltools.converters.mil import Builder as mb
from coremltools.converters.mil.mil import types as mil_types
from coremltools.converters.mil.mil.passes.pass_pipeline import PassPipeline


def rope_tables(max_seq, hd, base=500000.0):
    pos = np.arange(max_seq, dtype=np.float32)
    freqs = 1.0 / (base ** (np.arange(0, hd, 2, dtype=np.float32) / hd))
    return np.cos(np.outer(pos, freqs)).astype(np.float16), \
           np.sin(np.outer(pos, freqs)).astype(np.float16)


def rms_norm(x, w, eps, p):
    sq = mb.mul(x=x, y=x, name=f"{p}sq")
    m = mb.reduce_mean(x=sq, axes=[-1], keep_dims=True, name=f"{p}m")
    inv = mb.rsqrt(x=mb.add(x=m, y=eps, name=f"{p}e"), name=f"{p}i")
    return mb.mul(x=mb.mul(x=x, y=inv, name=f"{p}n"), y=w, name=f"{p}o")


def apply_rope(t, cos_p, sin_p, heads, hd, p):
    dp = hd // 2
    te = mb.slice_by_size(x=t, begin=[0,0,0,0], size=[1,int(heads),1,int(dp)], name=f"{p}e")
    to = mb.slice_by_size(x=t, begin=[0,0,0,int(dp)], size=[1,int(heads),1,int(dp)], name=f"{p}o")
    re = mb.sub(x=mb.mul(x=te, y=cos_p, name=f"{p}ec"), y=mb.mul(x=to, y=sin_p, name=f"{p}os"), name=f"{p}re")
    ro = mb.add(x=mb.mul(x=te, y=sin_p, name=f"{p}es"), y=mb.mul(x=to, y=cos_p, name=f"{p}oc"), name=f"{p}ro")
    return mb.concat(values=[re, ro], axis=3, name=f"{p}r")


def cache_update(cached, new_kv, off_sq, max_seq, p):
    oh = mb.one_hot(indices=mb.reshape(x=off_sq, shape=[1], name=f"{p}oi"),
                    one_hot_vector_size=int(max_seq), axis=0, name=f"{p}oh")
    mask = mb.cast(x=mb.reshape(x=oh, shape=[1,1,int(max_seq),1], name=f"{p}or"),
                   dtype="fp16", name=f"{p}mk")
    inv = mb.sub(x=np.float16(1.0), y=mask, name=f"{p}iv")
    return mb.add(x=mb.mul(x=cached, y=inv, name=f"{p}ol"),
                  y=mb.mul(x=new_kv, y=mask, name=f"{p}nw"), name=f"{p}u")


def build_layer(h, k_state, v_state, W, off_sq, off_p1, cos_p, sin_p, attn_mask,
                eps, D, H, KVH, hd, qDim, kvDim, rep, dp, I, max_seq, idx):
    """Build one transformer layer. Returns updated h."""
    p = f"L{idx}"

    norm1 = rms_norm(h, W["n1"], eps, f"{p}a")
    q = mb.matmul(x=norm1, y=W["wq"], transpose_y=True, name=f"{p}qp")
    k = mb.matmul(x=norm1, y=W["wk"], transpose_y=True, name=f"{p}kp")
    v = mb.matmul(x=norm1, y=W["wv"], transpose_y=True, name=f"{p}vp")

    q = mb.transpose(x=mb.reshape(x=q, shape=[1,1,int(H),int(hd)], name=f"{p}qr"), perm=[0,2,1,3], name=f"{p}qt")
    k = mb.transpose(x=mb.reshape(x=k, shape=[1,1,int(KVH),int(hd)], name=f"{p}kr"), perm=[0,2,1,3], name=f"{p}kt")
    v = mb.transpose(x=mb.reshape(x=v, shape=[1,1,int(KVH),int(hd)], name=f"{p}vr"), perm=[0,2,1,3], name=f"{p}vt")

    q = apply_rope(q, cos_p, sin_p, H, hd, f"{p}Q")
    k = apply_rope(k, cos_p, sin_p, KVH, hd, f"{p}K")

    ck = mb.read_state(input=k_state, name=f"{p}rk")
    cv = mb.read_state(input=v_state, name=f"{p}rv")
    uk = cache_update(ck, k, off_sq, max_seq, f"{p}ck")
    uv = cache_update(cv, v, off_sq, max_seq, f"{p}cv")
    mb.coreml_update_state(state=k_state, value=uk, name=f"{p}wk")
    mb.coreml_update_state(state=v_state, value=uv, name=f"{p}wv")

    ak, av = uk, uv
    if rep > 1:
        ak = mb.tile(x=uk, reps=[1,int(rep),1,1], name=f"{p}kr2")
        av = mb.tile(x=uv, reps=[1,int(rep),1,1], name=f"{p}vr2")

    attn = mb.scaled_dot_product_attention(query=q, key=ak, value=av, attn_mask=attn_mask, name=f"{p}sd")
    attn = mb.reshape(x=mb.transpose(x=attn, perm=[0,2,1,3], name=f"{p}at"), shape=[1,1,int(qDim)], name=f"{p}af")
    proj = mb.matmul(x=attn, y=W["wo"], transpose_y=True, name=f"{p}op")
    h = mb.add(x=h, y=proj, name=f"{p}ar")

    norm2 = rms_norm(h, W["n2"], eps, f"{p}m")
    gate = mb.matmul(x=norm2, y=W["wg"], transpose_y=True, name=f"{p}gp")
    up = mb.matmul(x=norm2, y=W["wu"], transpose_y=True, name=f"{p}up")
    act = mb.mul(x=mb.silu(x=gate, name=f"{p}sl"), y=up, name=f"{p}sw")
    down = mb.matmul(x=act, y=W["wd"], transpose_y=True, name=f"{p}dp")
    h = mb.add(x=h, y=down, name=f"{p}mr")
    return h


def generate(D, H, KVH, hd, I, L, max_seq, vocab, output_dir):
    qDim, kvDim, rep, dp = H*hd, KVH*hd, H//KVH, hd//2
    eps = np.float16(1e-5)
    np.random.seed(42)

    layer_weights = []
    for _ in range(L):
        layer_weights.append({
            "n1": np.ones(D, dtype=np.float16),
            "wq": (np.random.randn(qDim, D)*0.02).astype(np.float16),
            "wk": (np.random.randn(kvDim, D)*0.02).astype(np.float16),
            "wv": (np.random.randn(kvDim, D)*0.02).astype(np.float16),
            "wo": (np.random.randn(D, qDim)*0.02).astype(np.float16),
            "n2": np.ones(D, dtype=np.float16),
            "wg": (np.random.randn(I, D)*0.02).astype(np.float16),
            "wu": (np.random.randn(I, D)*0.02).astype(np.float16),
            "wd": (np.random.randn(D, I)*0.02).astype(np.float16),
        })
    w_emb = (np.random.randn(vocab, D)*0.02).astype(np.float16)
    w_fn = np.ones(D, dtype=np.float16)
    cos_t, sin_t = rope_tables(max_seq, hd)

    # Build input_specs
    input_specs = [
        mb.TensorSpec(shape=(1,1), dtype=mil_types.int32),
        mb.TensorSpec(shape=(1,), dtype=mil_types.int32),
    ]
    for i in range(L):
        input_specs.append(mb.StateTensorSpec(shape=(1,KVH,max_seq,hd), dtype=mil_types.fp16))
        input_specs.append(mb.StateTensorSpec(shape=(1,KVH,max_seq,hd), dtype=mil_types.fp16))

    # Dynamically create function with correct parameter count
    param_names = ["token_ids", "offset"]
    for i in range(L):
        param_names.extend([f"ks{i}", f"vs{i}"])

    func_params = ", ".join(param_names)

    # Build the function body as a string and exec it
    func_code = f"""
def _decoder({func_params}):
    h = mb.gather(x=w_emb, indices=token_ids, axis=0, name="emb")
    off_sq = mb.squeeze(x=offset, axes=[0], name="osq")
    off_p1 = mb.add(x=off_sq, y=np.int32(1), name="op1")
    cos_p = mb.reshape(x=mb.gather(x=cos_t, indices=off_sq, axis=0, name="cg"), shape=[1,1,1,{dp}], name="cr")
    sin_p = mb.reshape(x=mb.gather(x=sin_t, indices=off_sq, axis=0, name="sg"), shape=[1,1,1,{dp}], name="sr")
    seq_r = mb.range_1d(start=np.int32(0), end=np.int32({max_seq}), step=np.int32(1), name="sr2")
    valid = mb.less(x=seq_r, y=off_p1, name="vld")
    am = mb.mul(x=mb.sub(x=np.float16(1.0), y=mb.cast(x=mb.reshape(x=valid, shape=[1,1,1,{max_seq}], name="vr"), dtype="fp16", name="vf"), name="iv"), y=np.float16(-1e4), name="am")
"""
    for i in range(L):
        func_code += f"    h = build_layer(h, ks{i}, vs{i}, layer_weights[{i}], off_sq, off_p1, cos_p, sin_p, am, eps, {D}, {H}, {KVH}, {hd}, {qDim}, {kvDim}, {rep}, {dp}, {I}, {max_seq}, {i})\n"
    func_code += f"""    final = rms_norm(h, w_fn, eps, "fn")
    logits = mb.matmul(x=final, y=w_emb, transpose_y=True, name="lm")
    return logits
"""

    local_ns = {
        "mb": mb, "np": np, "w_emb": w_emb, "w_fn": w_fn, "cos_t": cos_t, "sin_t": sin_t,
        "eps": eps, "layer_weights": layer_weights, "build_layer": build_layer, "rms_norm": rms_norm,
    }
    exec(func_code, local_ns)
    decoder_fn = local_ns["_decoder"]

    pipeline = PassPipeline.DEFAULT
    pipeline.remove_passes({"common::canonicalize_inplace_pattern"})

    print(f"Building MIL program ({L} layers, D={D})...")
    prog = mb.program(input_specs=input_specs, opset_version=ct.target.iOS18)(decoder_fn)

    print(f"Converting to CoreML...")
    model = ct.convert(prog,
                       minimum_deployment_target=ct.target.macOS15,
                       compute_precision=ct.precision.FLOAT16,
                       pass_pipeline=pipeline)

    os.makedirs(output_dir, exist_ok=True)
    name = f"full_D{D}_L{L}_seq{max_seq}"
    path = os.path.join(output_dir, f"{name}.mlpackage")
    model.save(path)

    per_layer = D + qDim*D + kvDim*D*2 + D*qDim + D + I*D*2 + D*I
    total = vocab*D + per_layer*L + D
    kv = 2*L*KVH*max_seq*hd*2
    print(f"\nSaved: {path}")
    print(f"  D={D}, H={H}, KVH={KVH}, hd={hd}, I={I}, L={L}")
    print(f"  Weights: {total*2/1e6:.1f} MB, KV cache: {kv/1e6:.1f} MB")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", default="/tmp/full-coreml")
    p.add_argument("--D", type=int, default=896)
    p.add_argument("--H", type=int, default=14)
    p.add_argument("--KVH", type=int, default=2)
    p.add_argument("--hd", type=int, default=64)
    p.add_argument("--I", type=int, default=4864)
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--max-seq-len", type=int, default=256)
    p.add_argument("--vocab-size", type=int, default=32000)
    a = p.parse_args()
    generate(a.D, a.H, a.KVH, a.hd, a.I, a.layers, a.max_seq_len, a.vocab_size, a.output_dir)


if __name__ == "__main__":
    main()
