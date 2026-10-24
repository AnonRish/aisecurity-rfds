"""A small mixture-of-experts language model trained from scratch in numpy, so that its routing is learned.

The model predicts the next token from the current one. Each of ``layers`` blocks routes every token to its top
``k`` of ``experts`` small MLPs through a linear router, mixes their outputs with the softmax of the chosen router
logits, and adds the result to a residual stream. A per-position context vector, scaled by ``ctx``, is added to
the embedding: it stands in for everything else a real router sees (earlier tokens, position) and makes the route of a
token vary from one position to the next. Gradients are written by hand and checked against finite differences.

This is a toy: no attention, one token at a time, a synthetic language. It exists so that "which experts ran" is the
output of a trained router and so that "model quality" is a measured loss.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from aisecurity_rfds.common.rng import make_rng

WEIGHT_NAMES = ("emb", "Wr", "W1", "W2", "Wo")  # tensors that quantization rounds; the biases stay exact


def softmax(z: np.ndarray) -> np.ndarray:
    """Softmax over the last axis."""
    e = np.exp(z - z.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def top_k(scores: np.ndarray, k: int) -> np.ndarray:
    """Indices of the ``k`` largest entries of each row, in increasing index order."""
    return np.sort(np.argpartition(-scores, k - 1, axis=1)[:, :k], axis=1)


@dataclass(frozen=True)
class Config:
    vocab: int = 64
    dim: int = 24
    hidden: int = 24
    layers: int = 3
    experts: int = 16
    k: int = 2
    ctx: float = 0.5  # scale of the context vector added to the embedding

    def __post_init__(self) -> None:
        if min(self.vocab, self.dim, self.hidden, self.layers) < 1 or self.ctx < 0:
            raise ValueError("sizes must be positive and the context scale must not be negative")
        if not 1 <= self.k <= self.experts:
            raise ValueError("k must be between 1 and the number of experts")


def init_params(cfg: Config, seed: int) -> dict[str, np.ndarray]:
    rng = make_rng(seed)
    L, E, d, h = cfg.layers, cfg.experts, cfg.dim, cfg.hidden
    return {
        "emb": rng.normal(0.0, 1.0, (cfg.vocab, d)),
        "Wr": rng.normal(0.0, 1.0 / np.sqrt(d), (L, d, E)),
        "W1": rng.normal(0.0, np.sqrt(2.0 / d), (L, E, d, h)),
        "b1": np.zeros((L, E, h)),
        "W2": rng.normal(0.0, 0.5 / np.sqrt(h), (L, E, h, d)),
        "Wo": rng.normal(0.0, 1.0 / np.sqrt(d), (d, cfg.vocab)),
    }


def copy_params(p: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {name: value.copy() for name, value in p.items()}


# --- the synthetic language ------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Chain:
    """A first-order Markov chain over the vocabulary: ``trans[i]`` is the distribution of the token after ``i``."""

    trans: np.ndarray

    @classmethod
    def make(cls, seed: int, vocab: int = 64, latent: int = 6, width: int = 32, scale: float = 1.6) -> Chain:
        """Next-token distributions that are a nonlinear function of a few hidden features of each token, so that a
        model needs its experts to fit them."""
        rng = make_rng(seed)
        u = rng.normal(size=(vocab, latent))
        hidden = np.tanh(u @ rng.normal(size=(latent, width)))
        return cls(softmax(scale * hidden @ rng.normal(size=(width, vocab))))

    @property
    def entropy(self) -> float:
        """Mean conditional entropy in nats over uniformly drawn current tokens: the best loss any model can reach."""
        return float(-(self.trans * np.log(np.where(self.trans > 0, self.trans, 1.0))).sum(axis=1).mean())

    def next_tokens(self, tokens: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        cumulative = self.trans.cumsum(axis=1)[tokens]
        picked = (rng.random(len(tokens))[:, None] > cumulative).sum(axis=1)
        return np.minimum(picked, self.trans.shape[1] - 1)

    def mixed_with(self, other: Chain, share: float) -> Chain:
        """The chain whose rows are ``(1 - share)`` of this one's plus ``share`` of the other's."""
        return Chain((1.0 - share) * self.trans + share * other.trans)


# --- forward and backward --------------------------------------------------------------------------------------


def forward(
    cfg: Config,
    p: dict[str, np.ndarray],
    tokens: np.ndarray,
    c: np.ndarray,
    noise: np.ndarray | None = None,
    forced: np.ndarray | None = None,
    experts_on: bool = True,
):
    """Run a batch. Returns the logits, the routes ``(batch, layers, k)`` and what the backward pass needs.

    ``noise`` (batch, layers, experts) is added to the router logits *for the choice of experts only*: the mixing
    weights still come from the clean logits of the experts that were chosen. ``forced`` fixes the routes instead.
    With ``experts_on=False`` every block is the identity.
    """
    x = p["emb"][tokens] + cfg.ctx * c
    routes = np.empty((len(tokens), cfg.layers, cfg.k), dtype=np.int64)
    cache = []
    for layer in range(cfg.layers):
        z = x @ p["Wr"][layer]
        if forced is not None:
            top = forced[:, layer, :]
        else:
            top = top_k(z if noise is None else z + noise[:, layer, :], cfg.k)
        routes[:, layer] = top
        g = softmax(np.take_along_axis(z, top, axis=1))
        y = np.zeros_like(x)
        parts = []
        if experts_on:
            for e in range(cfg.experts):
                rows, slot = np.nonzero(top == e)
                if rows.size == 0:
                    continue
                h = np.maximum(x[rows] @ p["W1"][layer, e] + p["b1"][layer, e], 0.0)
                out = h @ p["W2"][layer, e]
                y[rows] += g[rows, slot][:, None] * out
                parts.append((e, rows, slot, h, out))
        cache.append((x, z, top, g, parts))
        x = x + y
    return x @ p["Wo"], routes, (cache, x)


def loss_and_grads(
    cfg: Config,
    p: dict[str, np.ndarray],
    tokens: np.ndarray,
    targets: np.ndarray,
    c: np.ndarray,
    noise: np.ndarray | None = None,
    forced: np.ndarray | None = None,
    alpha: float = 0.0,
    experts_on: bool = True,
):
    """Cross-entropy of the next token plus ``alpha`` times the load-balancing loss of the Switch Transformer, with
    gradients of the sum. Returns ``(total, cross_entropy, grads)``."""
    logits, _, (cache, x_last) = forward(cfg, p, tokens, c, noise, forced, experts_on)
    n = len(tokens)
    probs = softmax(logits)
    ce = float(-np.log(probs[np.arange(n), targets]).mean())
    dlogits = probs.copy()
    dlogits[np.arange(n), targets] -= 1.0
    dlogits /= n
    grads = {name: np.zeros_like(value) for name, value in p.items()}
    grads["Wo"] = x_last.T @ dlogits
    dx = dlogits @ p["Wo"].T
    aux = 0.0
    for layer in reversed(range(cfg.layers)):
        x_in, z, top, g, parts = cache[layer]
        dx_in = dx.copy()
        dg = np.zeros_like(g)
        for e, rows, slot, h, out in parts:
            dy = g[rows, slot][:, None] * dx[rows]
            dg[rows, slot] += (dx[rows] * out).sum(axis=1)
            grads["W2"][layer, e] += h.T @ dy
            dh = (dy @ p["W2"][layer, e].T) * (h > 0)
            grads["W1"][layer, e] += x_in[rows].T @ dh
            grads["b1"][layer, e] += dh.sum(axis=0)
            dx_in[rows] += dh @ p["W1"][layer, e].T
        dz = np.zeros_like(z)
        np.put_along_axis(dz, top, g * (dg - (g * dg).sum(axis=1, keepdims=True)), axis=1)
        if alpha > 0:
            pz = softmax(z)
            share = np.bincount(top.ravel(), minlength=cfg.experts) / (n * cfg.k)  # fraction of dispatches per expert
            aux += alpha * cfg.experts * float((share * pz.mean(axis=0)).sum())
            dz += (alpha * cfg.experts / n) * pz * (share[None, :] - (pz * share[None, :]).sum(axis=1, keepdims=True))
        grads["Wr"][layer] = x_in.T @ dz
        dx_in += dz @ p["Wr"][layer].T
        dx = dx_in
    np.add.at(grads["emb"], tokens, dx)
    return ce + aux, ce, grads


def routes_of(
    cfg: Config,
    p: dict[str, np.ndarray],
    tokens: np.ndarray,
    c: np.ndarray,
    noise: np.ndarray | None = None,
) -> np.ndarray:
    """The experts chosen for each token in each layer: ``(len(tokens), layers, k)``."""
    return forward(cfg, p, tokens, c, noise)[1]


def routing_noise(cfg: Config, n: int, tau: float, rng: np.random.Generator) -> np.ndarray | None:
    """Gumbel noise of scale ``tau`` for ``n`` tokens, or None when ``tau`` is zero.

    Adding it to router logits before the top-1 choice samples each expert with probability softmax(logits / tau)
    (the Gumbel-max trick); the top-``k`` choice samples without replacement."""
    if tau < 0:
        raise ValueError("tau must not be negative")
    return None if tau == 0 else tau * rng.gumbel(size=(n, cfg.layers, cfg.experts))


# --- training and evaluation -----------------------------------------------------------------------------------


def train(
    cfg: Config,
    params: dict[str, np.ndarray],
    chain: Chain,
    steps: int,
    rng: np.random.Generator,
    lr: float = 3e-3,
    batch: int = 256,
    alpha: float = 0.01,
    tau: float = 0.0,
    experts_on: bool = True,
) -> dict[str, np.ndarray]:
    """Adam on fresh uniformly drawn tokens and their successors; routing noise of scale ``tau`` is applied in training."""
    p = copy_params(params)
    m = {name: np.zeros_like(v) for name, v in p.items()}
    v = {name: np.zeros_like(x) for name, x in p.items()}
    b1, b2, eps = 0.9, 0.999, 1e-8
    for t in range(1, steps + 1):
        tokens = rng.integers(0, cfg.vocab, batch)
        targets = chain.next_tokens(tokens, rng)
        c = rng.normal(size=(batch, cfg.dim))
        noise = routing_noise(cfg, batch, tau, rng)
        _, _, grads = loss_and_grads(cfg, p, tokens, targets, c, noise, None, alpha, experts_on)
        for name in p:
            m[name] = b1 * m[name] + (1 - b1) * grads[name]
            v[name] = b2 * v[name] + (1 - b2) * grads[name] ** 2
            p[name] -= lr * (m[name] / (1 - b1**t)) / (np.sqrt(v[name] / (1 - b2**t)) + eps)
    return p


def evaluate(
    cfg: Config,
    p: dict[str, np.ndarray],
    chain: Chain,
    n: int,
    rng: np.random.Generator,
    tau: float = 0.0,
    experts_on: bool = True,
) -> float:
    """Mean cross-entropy (nats) of the next token on ``n`` fresh samples, with routing noise ``tau``."""
    tokens = rng.integers(0, cfg.vocab, n)
    targets = chain.next_tokens(tokens, rng)
    c = rng.normal(size=(n, cfg.dim))
    noise = routing_noise(cfg, n, tau, rng)
    logits = forward(cfg, p, tokens, c, noise, None, experts_on)[0]
    probs = softmax(logits)
    return float(-np.log(probs[np.arange(n), targets]).mean())


def expert_load(cfg: Config, routes: np.ndarray) -> np.ndarray:
    """Share of all dispatches that each expert receives, per layer: ``(layers, experts)``, rows summing to one."""
    out = np.zeros((cfg.layers, cfg.experts))
    for layer in range(cfg.layers):
        out[layer] = np.bincount(routes[:, layer].ravel(), minlength=cfg.experts) / routes[:, layer].size
    return out


# --- quantization ----------------------------------------------------------------------------------------------


def quantize(p: dict[str, np.ndarray], bits: int, names: tuple[str, ...] = WEIGHT_NAMES) -> dict[str, np.ndarray]:
    """Round the named tensors to ``bits``-bit signed integers with one scale per tensor (symmetric, round to nearest)."""
    if bits < 2:
        raise ValueError("at least two bits")
    levels = 2 ** (bits - 1) - 1
    out = copy_params(p)
    for name in names:
        w = p[name]
        scale = float(np.abs(w).max()) / levels
        out[name] = np.clip(np.round(w / scale), -levels, levels) * scale if scale > 0 else w.copy()
    return out
