"""Is this reimplementation actually LoRA, or merely LoRA-shaped?

A from-scratch implementation is worth nothing if it quietly differs from the
one the ecosystem runs.  So: build the same model twice, adapt one with
``lora/`` and one with HuggingFace ``peft``, force both adapters to hold the
*same* numbers, and check that they agree -- on the forward pass, on the
gradients, and after merging.

Agreement on the forward pass alone would be weak evidence: two implementations
can match at initialisation (where B=0 makes everything a no-op) and still
train differently.  The gradient check is the one that matters, because it
pins the backward path -- including whether the alpha/r scaling is applied
where we think it is.

Requires ``peft``; skipped cleanly if unavailable.
"""

import argparse
import json
import os
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lab.model import GPT, GPTConfig
from lora import LoRAConfig, apply_lora, merge_lora
from lora.layers import lora_modules

TARGETS = ["q_proj", "v_proj"]


def build_model(seed: int, device: str) -> GPT:
    torch.manual_seed(seed)
    cfg = GPTConfig(n_layer=2, n_head=4, d_model=128, block_size=64)
    return GPT(cfg).to(device).eval()


def peft_lora_modules(model):
    """Yield peft LoRA layers regardless of container naming."""
    for name, mod in model.named_modules():
        if hasattr(mod, "lora_A") and hasattr(mod, "lora_B") and hasattr(mod, "base_layer"):
            yield name, mod


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/kaggle/working/results/05_peft_parity.json")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--r", type=int, default=8)
    ap.add_argument("--alpha", type=float, default=16.0)
    args = ap.parse_args()
    dev = args.device
    out: dict = {"r": args.r, "alpha": args.alpha, "targets": TARGETS}

    try:
        import peft
        from peft import LoraConfig as PeftLoraConfig, inject_adapter_in_model
        out["peft_version"] = peft.__version__
        # Environment workaround, not a LoRA detail: peft's quantisation
        # dispatcher calls is_torchao_available(), which *raises* rather than
        # returning False when an out-of-range torchao is installed (0.10 here).
        # Adapting a plain nn.Linear never touches that path, so stub the probe.
        import peft.import_utils as _piu
        _piu.is_torchao_available = lambda: False
        try:
            import peft.tuners.lora.torchao as _plt
            _plt.is_torchao_available = lambda: False
        except Exception:
            pass
    except Exception as e:  # pragma: no cover - environment dependent
        out["skipped"] = f"peft unavailable: {e}"
        print(out["skipped"], flush=True)
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        json.dump(out, open(args.out, "w"), indent=2)
        return

    ours = build_model(0, dev)
    theirs = build_model(0, dev)
    # Identical starting weights is a precondition, not an assumption.
    for (na, pa), (nb, pb) in zip(ours.named_parameters(), theirs.named_parameters()):
        assert torch.equal(pa, pb), f"base weights differ at {na}"

    apply_lora(ours, LoRAConfig(target_modules=tuple(TARGETS), r=args.r, alpha=args.alpha))
    theirs = inject_adapter_in_model(
        PeftLoraConfig(r=args.r, lora_alpha=args.alpha, target_modules=TARGETS,
                       lora_dropout=0.0, bias="none"),
        theirs,
    )

    ours_layers = dict(lora_modules(ours))
    peft_layers = dict(peft_lora_modules(theirs))
    out["n_adapted_ours"] = len(ours_layers)
    out["n_adapted_peft"] = len(peft_layers)
    print(f"adapted: ours={len(ours_layers)} peft={len(peft_layers)}", flush=True)

    # Copy our (randomly initialised) A into peft, and give B a non-zero value
    # in both -- at B=0 the adapter is a no-op and the test proves nothing.
    with torch.no_grad():
        for name, om in ours_layers.items():
            pm = peft_layers[name]
            nn.init.normal_(om.lora_B, std=0.05)
            pa = pm.lora_A["default"].weight
            pb = pm.lora_B["default"].weight
            assert pa.shape == om.lora_A.shape, f"A shape {pa.shape} vs {om.lora_A.shape}"
            assert pb.shape == om.lora_B.shape, f"B shape {pb.shape} vs {om.lora_B.shape}"
            pa.copy_(om.lora_A)
            pb.copy_(om.lora_B)
            out.setdefault("scaling", {})["ours"] = om.scaling
            out["scaling"]["peft"] = pm.scaling["default"]

    x = torch.randint(0, 256, (2, 32), device=dev)
    y = torch.randint(0, 256, (2, 32), device=dev)

    # ---- forward parity ---------------------------------------------------
    with torch.no_grad():
        lo, _ = ours(x)
        lp, _ = theirs(x)
    fwd = (lo - lp).abs().max().item()
    out["max_abs_logit_diff"] = fwd
    out["rel_logit_diff"] = fwd / lo.abs().max().item()
    print(f"[forward] max |diff| = {fwd:.3e} "
          f"(relative {out['rel_logit_diff']:.3e})", flush=True)

    # ---- gradient parity --------------------------------------------------
    ours.zero_grad()
    theirs.zero_grad()
    ours.train()
    theirs.train()
    _, loss_o = ours(x, y)
    _, loss_p = theirs(x, y)
    loss_o.backward()
    loss_p.backward()

    gdiff_a = gdiff_b = 0.0
    for name, om in ours_layers.items():
        pm = peft_layers[name]
        gdiff_a = max(gdiff_a,
                      (om.lora_A.grad - pm.lora_A["default"].weight.grad).abs().max().item())
        gdiff_b = max(gdiff_b,
                      (om.lora_B.grad - pm.lora_B["default"].weight.grad).abs().max().item())
    out["loss_ours"] = loss_o.item()
    out["loss_peft"] = loss_p.item()
    out["max_abs_grad_diff_A"] = gdiff_a
    out["max_abs_grad_diff_B"] = gdiff_b
    print(f"[loss]     ours {loss_o.item():.8f}  peft {loss_p.item():.8f}", flush=True)
    print(f"[backward] max |grad diff| A={gdiff_a:.3e} B={gdiff_b:.3e}", flush=True)

    # ---- merged parity ----------------------------------------------------
    ours.eval()
    theirs.eval()
    with torch.no_grad():
        merge_lora(ours)
        merged_ours, _ = ours(x)
        try:
            theirs.merge_adapter()
            merged_peft, _ = theirs(x)
            out["max_abs_merged_diff"] = (merged_ours - merged_peft).abs().max().item()
        except Exception as e:
            out["merge_note"] = f"peft merge_adapter unavailable: {e}"
            out["max_abs_merged_diff"] = (merged_ours - lp).abs().max().item()
    print(f"[merged]   max |diff| = {out['max_abs_merged_diff']:.3e}", flush=True)

    tol = 1e-4
    out["parity_ok"] = bool(
        out["rel_logit_diff"] < tol and gdiff_a < tol and gdiff_b < tol
        and out["max_abs_merged_diff"] < 1e-3
    )
    print(f"\nPARITY: {'PASS' if out['parity_ok'] else 'FAIL'}", flush=True)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=2)
    print(f"[saved] {args.out}", flush=True)


if __name__ == "__main__":
    main()
