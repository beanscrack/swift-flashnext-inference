p = "/u701/j692wu/swift-opt/mtp/residual/mtp_torch.py"
s = open(p).read()
old = """        probs = torch.softmax(scores, dim=-1)
        out = torch.einsum("bhts,bshd->bthd", probs, Vr)
        out = out * torch.sigmoid(gate.float())  # qsa.py:205 normalized_output * sigmoid(output_gate)
        out = out.to(x.dtype).reshape(B, T, self.nh * self.hd)
"""
new = """        if ATTN_MODE == "fp32":
            probs = torch.softmax(scores, dim=-1)
            out = torch.einsum("bhts,bshd->bthd", probs, Vr)
        else:
            # mirror ops/qsa.py _qsa_sparse_paged_gqa_splitk_kernel: P rounded to bf16 for the PV dot,
            # normalizer summed from fp32 P (ops/qsa.py:173-186)
            m = scores.amax(dim=-1, keepdim=True)
            pe = torch.exp(scores - m)
            denom = pe.sum(dim=-1, keepdim=True)
            num = torch.einsum("bhts,bshd->bthd", pe.to(torch.bfloat16).float(), Vr)
            out = num / denom.transpose(1, 2)
            if ATTN_MODE == "vllm":
                # ops/qsa.py:202-205 normalized output rounded to bf16 before the gate multiply
                out = out.to(torch.bfloat16).float()
        out = out * torch.sigmoid(gate.float())  # qsa.py:205 normalized_output * sigmoid(output_gate)
        out = out.to(x.dtype).reshape(B, T, self.nh * self.hd)
"""
assert s.count(old) == 1, s.count(old)
s = s.replace(old, new)
anchor = "MTP_PREFIX = "
assert s.count(anchor) == 1
i = s.index(anchor); j = s.index(chr(10), i) + 1
s = s[:j] + "ATTN_MODE = \"fp32\"  # residual experiments: fp32 | pbf16 | vllm\n" + s[j:]
open(p, "w").write(s)
print("patched")
