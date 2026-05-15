"""
Blackjack AI Advisor — standalone inference app.

Requirements: streamlit torch numpy
Run with:    streamlit run app.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import streamlit as st
import torch
import torch.nn as nn


# ─── Model (self-contained, no project imports needed) ────────────────────────

class _DuelingNet(nn.Module):
    """Minimal Dueling Q-Network for inference only."""

    def __init__(self, input_dim: int, output_dim: int, hidden_dims: list[int]) -> None:
        super().__init__()
        shared: list[nn.Module] = []
        prev = input_dim
        for h in hidden_dims[:-1]:
            shared += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        self.shared = nn.Sequential(*shared) if shared else nn.Identity()

        stream_in = hidden_dims[-2] if len(hidden_dims) > 1 else input_dim
        mid = hidden_dims[-1] // 2
        self.value_stream = nn.Sequential(nn.Linear(stream_in, mid), nn.ReLU(), nn.Linear(mid, 1))
        self.advantage_stream = nn.Sequential(nn.Linear(stream_in, mid), nn.ReLU(), nn.Linear(mid, output_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f = self.shared(x)
        v = self.value_stream(f)
        a = self.advantage_stream(f)
        return v + (a - a.mean(dim=1, keepdim=True))


@st.cache_resource(show_spinner="Loading model…")
def _load_net() -> tuple[_DuelingNet, int, int] | None:
    """Load the exported model.pt from the same directory as this script."""
    model_path = Path(__file__).parent / "model.pt"
    if not model_path.exists():
        return None
    ckpt = torch.load(model_path, map_location="cpu", weights_only=True)
    net = _DuelingNet(ckpt["input_dim"], ckpt["output_dim"], ckpt["hidden_dims"])
    net.load_state_dict(ckpt["weights"])
    net.eval()
    return net, ckpt["input_dim"], ckpt["output_dim"]


def _predict(net: _DuelingNet, player_sum: int, dealer_card: int, usable_ace: bool) -> tuple[str, str]:
    obs = torch.FloatTensor([player_sum / 31.0, dealer_card / 10.0, float(usable_ace)]).unsqueeze(0)
    with torch.no_grad():
        q = net(obs).numpy()[0]
    action = int(np.argmax(q))
    margin = abs(float(q[1] - q[0]))
    confidence = "High" if margin > 0.25 else "Medium" if margin > 0.08 else "Close call"
    return ("HIT" if action == 1 else "STAND"), confidence


# ─── Card helpers ─────────────────────────────────────────────────────────────

CARD_LABELS = ["A", "2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K"]
LABEL_TO_VALUE = {"A": 1, "2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7,
                  "8": 8, "9": 9, "10": 10, "J": 10, "Q": 10, "K": 10}
VALUE_LABEL = {1: "A", 2: "2", 3: "3", 4: "4", 5: "5", 6: "6",
               7: "7", 8: "8", 9: "9", 10: "10"}


def _hand_value(cards: list[int]) -> tuple[int, bool]:
    total = sum(cards)
    usable_ace = 1 in cards and total + 10 <= 21
    return total + (10 if usable_ace else 0), usable_ace


def _hi_lo(cards: list[int]) -> int:
    return sum(1 if 2 <= c <= 6 else -1 if c in (1, 10) else 0 for c in cards)


def _tips(cards: list[int], player_sum: int, dealer: int) -> list[str]:
    out = []
    if len(cards) == 2:
        c1, c2 = cards
        if c1 == c2 == 1:
            out.append("✂️ **Always Split Aces** — restart each hand with a powerful Ace.")
        elif c1 == c2 == 8:
            out.append("✂️ **Always Split 8s** — hard 16 is the worst hand; split it.")
        elif c1 == c2 == 10:
            out.append("🚫 **Never Split 10s** — you have 20, one of the best hands.")
        if player_sum == 11:
            out.append("⬆️ **Consider Doubling Down** — 11 is the best double opportunity.")
        elif player_sum == 10 and dealer <= 9:
            out.append(f"⬆️ **Consider Doubling Down** — 10 vs dealer {VALUE_LABEL.get(dealer, dealer)} is strong.")
        elif player_sum == 9 and 3 <= dealer <= 6:
            out.append("⬆️ **Consider Doubling Down** — 9 vs weak dealer (3–6) is a marginal double.")
    if player_sum == 16 and dealer in (9, 10, 1) and len(cards) == 2:
        out.append("🏳️ **Consider Surrender** (if the table allows it) — hard 16 vs 9/10/A saves half your bet.")
    if player_sum == 15 and dealer == 10 and len(cards) == 2:
        out.append("🏳️ **Consider Surrender** — hard 15 vs dealer 10 is the other classic surrender spot.")
    return out


# ─── Session state ────────────────────────────────────────────────────────────

def _init() -> None:
    for k, v in {"player_cards": [], "dealer_value": None,
                 "s_wins": 0, "s_losses": 0, "s_draws": 0, "cards_seen": []}.items():
        if k not in st.session_state:
            st.session_state[k] = v


# ─── App ──────────────────────────────────────────────────────────────────────

def main() -> None:
    st.set_page_config(page_title="Blackjack AI", page_icon="🃏", layout="centered",
                       initial_sidebar_state="collapsed")
    st.markdown("""
    <style>
    div[data-testid="column"] > div > div > div > button {
        font-size:1.1em !important; font-weight:bold !important; padding:0.4rem 0.1rem !important;
    }
    .abox { text-align:center; padding:1.2rem; border-radius:14px;
            font-size:2.6em; font-weight:900; letter-spacing:2px; margin:0.6rem 0; }
    .hit   { background:#ff4b4b; color:#fff; }
    .stand { background:#21c354; color:#fff; }
    .bust  { background:#808495; color:#fff; }
    .bj    { background:#ffd700; color:#1a1a1a; }
    </style>
    """, unsafe_allow_html=True)

    _init()
    st.title("🃏 Blackjack AI Advisor")

    result = _load_net()
    if result is None:
        st.error("Model file `model.pt` not found in this folder. "
                 "Ask your friend to send you the `model.pt` file and place it here.")
        st.stop()
    net, _, _ = result

    # ── YOUR HAND ──────────────────────────────────────────────────────────────
    st.subheader("Your Hand")
    st.caption("Tap each card as you receive it")
    cols = st.columns(len(CARD_LABELS))
    for i, lbl in enumerate(CARD_LABELS):
        if cols[i].button(lbl, key=f"p{lbl}"):
            st.session_state.player_cards.append(LABEL_TO_VALUE[lbl])
            st.rerun()

    if st.session_state.player_cards:
        total, soft = _hand_value(st.session_state.player_cards)
        readable = " + ".join(VALUE_LABEL.get(c, str(c)) for c in st.session_state.player_cards)
        st.markdown(f"### {total} ({'Soft' if soft else 'Hard'})  —  {readable}")
        if st.button("↩ Remove last card"):
            st.session_state.player_cards.pop()
            st.rerun()
    else:
        st.markdown("*No cards yet — tap above*")

    # ── DEALER CARD ────────────────────────────────────────────────────────────
    st.subheader("Dealer's Up Card")
    choice = st.selectbox("Dealer", ["— select —"] + CARD_LABELS, label_visibility="collapsed")
    st.session_state.dealer_value = LABEL_TO_VALUE[choice] if choice != "— select —" else None

    # ── RECOMMENDATION ─────────────────────────────────────────────────────────
    st.divider()
    pc = st.session_state.player_cards
    dv = st.session_state.dealer_value

    if pc and dv:
        total, soft = _hand_value(pc)

        if total > 21:
            st.markdown('<div class="abox bust">BUST 💀</div>', unsafe_allow_html=True)
        elif total == 21 and len(pc) == 2:
            st.markdown('<div class="abox bj">BLACKJACK ⭐</div>', unsafe_allow_html=True)
            st.info("Natural Blackjack! You win 1.5× unless the dealer also has 21.")
        else:
            action, confidence = _predict(net, total, dv, soft)
            css = "hit" if action == "HIT" else "stand"
            emoji = "🎯" if action == "HIT" else "✋"
            st.markdown(f'<div class="abox {css}">{action} {emoji}</div>', unsafe_allow_html=True)
            icons = {"High": "🟢", "Medium": "🟡", "Close call": "🔴"}
            st.caption(f"{icons[confidence]} Confidence: **{confidence}**")
            for tip in _tips(pc, total, dv):
                st.info(tip)
    elif pc:
        st.info("Select the dealer's card to get a recommendation")
    elif dv:
        st.info("Add your cards to get a recommendation")
    else:
        st.markdown("*Add your cards and the dealer's card above*")

    # ── CARD COUNT ─────────────────────────────────────────────────────────────
    with st.expander("📊 Hi-Lo Card Count", expanded=False):
        visible = list(pc) + ([dv] if dv else [])
        count = _hi_lo(st.session_state.cards_seen + visible)
        sign = "+" if count > 0 else ""
        if count > 2:
            st.success(f"Count: **{sign}{count}** — Deck favors the player 🎉")
        elif count < -2:
            st.warning(f"Count: **{count}** — Deck favors the dealer ⚠️")
        else:
            st.info(f"Count: **{sign}{count}** — Neutral")
        st.caption("2–6 = **+1**  |  7–9 = **0**  |  10/J/Q/K/A = **−1**")
        if count > 3:
            st.caption("💰 Bet more than usual")
        elif count < -3:
            st.caption("💰 Bet your minimum")

    # ── CONTROLS ───────────────────────────────────────────────────────────────
    st.divider()
    col_r, col_n = st.columns([3, 1])
    with col_r:
        opts = ["Record outcome…", "Win ✅", "Loss ❌", "Draw / Push 🤝"]
        res = st.selectbox("Outcome", opts, label_visibility="collapsed")
        if res != opts[0] and st.button("Save & next hand", type="primary"):
            hand = list(pc) + ([dv] if dv else [])
            st.session_state.cards_seen.extend(hand)
            if "Win" in res:
                st.session_state.s_wins += 1
            elif "Loss" in res:
                st.session_state.s_losses += 1
            else:
                st.session_state.s_draws += 1
            st.session_state.player_cards = []
            st.session_state.dealer_value = None
            st.rerun()
    with col_n:
        if st.button("🔄 New hand", use_container_width=True):
            st.session_state.player_cards = []
            st.session_state.dealer_value = None
            st.rerun()

    # ── STATS ──────────────────────────────────────────────────────────────────
    w, l, d = st.session_state.s_wins, st.session_state.s_losses, st.session_state.s_draws
    total = w + l + d
    if total > 0:
        st.divider()
        st.subheader("Session Stats")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Hands", total)
        c2.metric("Wins", w, f"{w/total*100:.0f}%")
        c3.metric("Losses", l)
        c4.metric("Draws", d)
        net_units = w - l
        st.caption(f"Net: **{'+' if net_units >= 0 else ''}{net_units}** units")
        if st.button("Reset stats"):
            st.session_state.s_wins = 0
            st.session_state.s_losses = 0
            st.session_state.s_draws = 0
            st.session_state.cards_seen = []
            st.rerun()


if __name__ == "__main__":
    main()
