"""Torch tests on toy tensors / a mock decoder (no model download, CPU OK)."""
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from trialmatch.attention import reference_attention, tiled_attention, torch_sdpa
from trialmatch.speculative import speculative_greedy


@pytest.mark.parametrize("B,H,Tq,Tk,D,off,bq,bk", [
    (1, 1, 1, 1, 8, 0, 4, 4), (1, 2, 7, 7, 16, 0, 3, 2), (2, 4, 64, 64, 32, 0, 16, 16),
    (1, 3, 1, 50, 16, 49, 8, 8), (1, 3, 5, 50, 16, 45, 4, 8), (1, 2, 33, 100, 16, 67, 8, 16)])
def test_tiled_matches_reference(B, H, Tq, Tk, D, off, bq, bk):
    torch.manual_seed(0)
    q, k, v = torch.randn(B, H, Tq, D) * 3, torch.randn(B, H, Tk, D) * 3, torch.randn(B, H, Tk, D)
    ref = reference_attention(q.double(), k.double(), v.double(), True, off)
    til = tiled_attention(q.double(), k.double(), v.double(), True, off, block_q=bq, block_k=bk)
    assert (til.double() - ref).abs().max() < 1e-5
    sd = torch_sdpa(q, k, v, True, off)
    assert (sd.double() - ref).abs().max() < 1e-4


class MockCache:
    def __init__(self):
        self.toks = []

    @property
    def length(self):
        return len(self.toks)

    def crop(self, n):
        self.toks = self.toks[:n]


class MockDecoder:
    """Next token is a deterministic function of the WHOLE prefix stored in the cache, so a
    desynchronised cache produces different tokens and the test fails."""
    V = 37

    def __init__(self, noise=0.0, seed=0):
        self.noise, self.device, self.attn_name = noise, "cpu", "mock"
        self.g = torch.Generator().manual_seed(seed)

    def new_cache(self, b, cap):
        return MockCache()

    def nxt(self, prefix):
        t = (sum(prefix) * 31 + len(prefix) * 7) % self.V
        if self.noise and torch.rand(1, generator=self.g).item() < self.noise:
            t = (t + 1) % self.V
        return t

    def forward(self, ids, cache=None, last_only=False):
        rows = []
        for t in ids[0].tolist():
            cache.toks.append(t)
            rows.append(torch.nn.functional.one_hot(torch.tensor(self.nxt(cache.toks)), self.V).float())
        out = torch.stack(rows)[None]
        return out[:, -1:] if last_only else out


def greedy(prompt, n, stop=None):
    s, out = list(prompt), []
    m = MockDecoder()
    for _ in range(n):
        t = m.nxt(s); out.append(t); s.append(t)
        if t == stop:
            break
    return out


@pytest.mark.parametrize("noise", [0.0, 0.3, 0.9])
@pytest.mark.parametrize("k", [1, 3, 5])
def test_speculative_equals_greedy(noise, k):
    import random
    random.seed(k)
    for _ in range(40):
        prompt = [random.randrange(37) for _ in range(random.randint(1, 6))]
        n = random.randint(1, 25)
        stop = random.choice([None, 5])
        r = speculative_greedy(MockDecoder(), MockDecoder(noise, seed=k), torch.tensor([prompt]), n, k,
                               stops=[stop] if stop is not None else None)
        assert r["tokens"] == greedy(prompt, n, stop)
        if noise == 0.0:
            assert r["acceptance_rate"] in (None, 1.0) or r["acceptance_rate"] > 0.99
