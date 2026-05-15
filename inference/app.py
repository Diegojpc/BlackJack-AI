"""
Blackjack AI Advisor — full button-only interface.

Requirements: streamlit torch numpy
Run with:    streamlit run app.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import streamlit as st
import torch
import torch.nn as nn


# ─── Neural network (self-contained, no project imports) ──────────────────────

class _DuelingNet(nn.Module):
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
        v, a = self.value_stream(f), self.advantage_stream(f)
        return v + (a - a.mean(dim=1, keepdim=True))


@st.cache_resource(show_spinner="Loading AI…")
def _load_net() -> _DuelingNet | None:
    path = Path(__file__).parent / "model.pt"
    if not path.exists():
        return None
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    net = _DuelingNet(ckpt["input_dim"], ckpt["output_dim"], ckpt["hidden_dims"])
    net.load_state_dict(ckpt["weights"])
    net.eval()
    return net


def _predict(net: _DuelingNet, player_sum: int, dealer_card: int, usable_ace: bool) -> tuple[str, str]:
    obs = torch.FloatTensor([player_sum / 31.0, dealer_card / 10.0, float(usable_ace)]).unsqueeze(0)
    with torch.no_grad():
        q = net(obs).numpy()[0]
    margin = abs(float(q[1] - q[0]))
    return (
        "HIT" if int(np.argmax(q)) == 1 else "STAND",
        "High" if margin > 0.25 else "Medium" if margin > 0.08 else "Close call",
    )


# ─── Card helpers ─────────────────────────────────────────────────────────────

ROW1 = ["A", "2", "3", "4", "5", "6", "7"]
ROW2 = ["8", "9", "10", "J", "Q", "K"]
ALL_LABELS = ROW1 + ROW2

LABEL_TO_VALUE = {
    "A": 1, "2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7,
    "8": 8, "9": 9, "10": 10, "J": 10, "Q": 10, "K": 10,
}
VALUE_TO_SHORT = {1: "A", 2: "2", 3: "3", 4: "4", 5: "5", 6: "6",
                  7: "7", 8: "8", 9: "9", 10: "10"}


def _hand_value(cards: list[int]) -> tuple[int, bool]:
    total = sum(cards)
    soft = 1 in cards and total + 10 <= 21
    return total + (10 if soft else 0), soft


def _hi_lo(cards: list[int]) -> int:
    return sum(1 if 2 <= c <= 6 else -1 if c in (1, 10) else 0 for c in cards)


def _readable(cards: list[int]) -> str:
    return " + ".join(VALUE_TO_SHORT.get(c, str(c)) for c in cards)


def _tips(cards: list[int], player_sum: int, dealer: int) -> list[str]:
    tips: list[str] = []
    if len(cards) == 2:
        c1, c2 = cards
        if c1 == c2 == 1:
            tips.append("✂️ **Always Split Aces** — each hand restarts with a powerful Ace.")
        elif c1 == c2 == 8:
            tips.append("✂️ **Always Split 8s** — hard 16 is the worst hand; two 8s gives better odds.")
        elif c1 == c2 == 10:
            tips.append("🚫 **Never Split 10s** — a 20 is one of the strongest hands.")
        if player_sum == 11:
            tips.append("⬆️ **Consider Doubling Down** — 11 is your best doubling opportunity.")
        elif player_sum == 10 and dealer <= 9:
            tips.append(f"⬆️ **Consider Doubling Down** — 10 vs dealer {VALUE_TO_SHORT.get(dealer, dealer)} is strong.")
        elif player_sum == 9 and 3 <= dealer <= 6:
            tips.append("⬆️ **Consider Doubling Down** — 9 vs weak dealer (3–6) is a marginal double.")
    if player_sum == 16 and dealer in (9, 10, 1) and len(cards) == 2:
        tips.append("🏳️ **Consider Surrender** — hard 16 vs 9/10/A saves half your bet if the table allows it.")
    if player_sum == 15 and dealer == 10 and len(cards) == 2:
        tips.append("🏳️ **Consider Surrender** — hard 15 vs dealer 10 is the other classic spot.")
    return tips


# ─── Session state ────────────────────────────────────────────────────────────

def _init() -> None:
    defaults: dict = {
        "player_cards": [],
        "dealer_value": None,
        "s_wins": 0,
        "s_losses": 0,
        "s_draws": 0,
        "cards_seen": [],
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def _record_and_reset(outcome: str) -> None:
    """Save outcome, extend card count history, clear current hand."""
    seen = list(st.session_state.player_cards)
    if st.session_state.dealer_value:
        seen.append(st.session_state.dealer_value)
    st.session_state.cards_seen.extend(seen)
    if outcome == "win":
        st.session_state.s_wins += 1
    elif outcome == "loss":
        st.session_state.s_losses += 1
    else:
        st.session_state.s_draws += 1
    st.session_state.player_cards = []
    st.session_state.dealer_value = None
    st.rerun()


# ─── Reusable card-button grid ────────────────────────────────────────────────

def _card_grid(prefix: str, highlight_value: int | None = None) -> str | None:
    """
    Render two rows of card buttons.
    Returns the label of the clicked card, or None.
    highlight_value: if set, that card's button shows as primary (dealer selection).
    """
    clicked: str | None = None
    for row in (ROW1, ROW2):
        cols = st.columns(len(row))
        for i, lbl in enumerate(row):
            val = LABEL_TO_VALUE[lbl]
            is_selected = highlight_value is not None and val == highlight_value and lbl == VALUE_TO_SHORT.get(val, lbl)
            # For 10/J/Q/K all map to value 10 — highlight whichever was clicked
            btn_type = "primary" if is_selected else "secondary"
            if cols[i].button(lbl, key=f"{prefix}_{lbl}", use_container_width=True, type=btn_type):
                clicked = lbl
    return clicked


# ─── CSS ──────────────────────────────────────────────────────────────────────

CSS = """
<style>
/* Bigger card buttons */
button[data-testid="baseButton-secondary"],
button[data-testid="baseButton-primary"] {
    font-size: 1.1em !important;
    font-weight: 700 !important;
    padding: 0.45rem 0.1rem !important;
    border-radius: 8px !important;
}

/* Action recommendation box */
.abox {
    text-align: center;
    padding: 1.4rem 1rem;
    border-radius: 16px;
    font-size: 2.8em;
    font-weight: 900;
    letter-spacing: 3px;
    margin: 0.5rem 0 0.3rem 0;
}
.hit       { background: #e63946; color: #fff; }
.stand     { background: #2dc653; color: #fff; }
.bust      { background: #6c757d; color: #fff; }
.blackjack { background: #f4c430; color: #1a1a1a; }

/* Outcome buttons */
div[data-outcome="win"] button   { background-color: #2dc653 !important; color: #fff !important; border: none !important; font-size: 1.2em !important; font-weight: 700 !important; }
div[data-outcome="loss"] button  { background-color: #e63946 !important; color: #fff !important; border: none !important; font-size: 1.2em !important; font-weight: 700 !important; }
div[data-outcome="draw"] button  { background-color: #6c757d !important; color: #fff !important; border: none !important; font-size: 1.2em !important; font-weight: 700 !important; }

/* Card display box */
.card-box {
    display: inline-block;
    font-size: 1.6em;
    font-weight: 900;
    border: 3px solid currentColor;
    border-radius: 10px;
    padding: 0.2rem 0.6rem;
    min-width: 2.5rem;
    text-align: center;
}
.section-label {
    font-size: 0.85em;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 1px;
    color: #888;
    margin: 0.8rem 0 0.3rem 0;
}
</style>
"""


# ─── App ──────────────────────────────────────────────────────────────────────

def main() -> None:
    st.set_page_config(
        page_title="Blackjack AI",
        page_icon="🃏",
        layout="centered",
        initial_sidebar_state="collapsed",
    )
    st.markdown(CSS, unsafe_allow_html=True)
    _init()

    net = _load_net()
    if net is None:
        st.error("model.pt not found — place it in the same folder as app.py")
        st.stop()

    st.title("🃏 Blackjack AI")

    # ── YOUR CARDS ─────────────────────────────────────────────────────────────
    st.markdown('<p class="section-label">Your cards — tap each card you receive</p>',
                unsafe_allow_html=True)

    clicked_player = _card_grid("p")
    if clicked_player:
        st.session_state.player_cards.append(LABEL_TO_VALUE[clicked_player])
        st.rerun()

    # Hand summary
    pc = st.session_state.player_cards
    if pc:
        total, soft = _hand_value(pc)
        hand_type = "Soft" if soft else "Hard"
        col_sum, col_undo = st.columns([3, 1])
        with col_sum:
            cards_html = "  ".join(
                f'<span class="card-box">{VALUE_TO_SHORT.get(c, c)}</span>'
                for c in pc
            )
            st.markdown(
                f"<div style='margin:0.4rem 0'>{cards_html}</div>"
                f"<div style='font-size:1.5em; font-weight:700; margin-top:0.2rem'>"
                f"  {total} <span style='font-size:0.65em; color:#888'>({hand_type})</span>"
                f"</div>",
                unsafe_allow_html=True,
            )
        with col_undo:
            st.markdown("<div style='margin-top:0.8rem'></div>", unsafe_allow_html=True)
            if st.button("↩ Undo", use_container_width=True):
                pc.pop()
                st.rerun()
    else:
        st.markdown(
            "<p style='color:#aaa; margin:0.5rem 0'>No cards yet</p>",
            unsafe_allow_html=True,
        )

    st.divider()

    # ── DEALER'S CARD ──────────────────────────────────────────────────────────
    st.markdown('<p class="section-label">Dealer\'s face-up card — tap to select</p>',
                unsafe_allow_html=True)

    # Determine which label is currently selected (for primary highlight)
    dv = st.session_state.dealer_value
    # Map stored value back to a label for the highlight logic
    selected_label = VALUE_TO_SHORT.get(dv) if dv else None

    clicked_dealer = _card_grid("d", highlight_value=dv)
    if clicked_dealer:
        st.session_state.dealer_value = LABEL_TO_VALUE[clicked_dealer]
        st.rerun()

    # Show selected dealer card prominently, or prompt
    if dv:
        label = VALUE_TO_SHORT.get(dv, str(dv))
        col_card, col_clear = st.columns([3, 1])
        with col_card:
            st.markdown(
                f"<div style='margin:0.4rem 0'>"
                f"<span class='card-box' style='color:#1f77b4'>{label}</span>"
                f"  <span style='color:#888; font-size:0.9em'>Dealer shows</span>"
                f"</div>",
                unsafe_allow_html=True,
            )
        with col_clear:
            st.markdown("<div style='margin-top:0.3rem'></div>", unsafe_allow_html=True)
            if st.button("✕ Clear", use_container_width=True):
                st.session_state.dealer_value = None
                st.rerun()
    else:
        st.markdown(
            "<p style='color:#aaa; margin:0.4rem 0'>No card selected</p>",
            unsafe_allow_html=True,
        )

    st.divider()

    # ── AI RECOMMENDATION ──────────────────────────────────────────────────────
    dv = st.session_state.dealer_value  # re-read after possible update

    if pc and dv:
        total, soft = _hand_value(pc)

        if total > 21:
            st.markdown('<div class="abox bust">BUST 💀</div>', unsafe_allow_html=True)

        elif total == 21 and len(pc) == 2:
            st.markdown('<div class="abox blackjack">BLACKJACK ⭐</div>', unsafe_allow_html=True)
            st.success("Natural 21 — you win 1.5× your bet unless dealer also has blackjack.")

        else:
            action, confidence = _predict(net, total, dv, soft)
            css = "hit" if action == "HIT" else "stand"
            icon = "🎯" if action == "HIT" else "✋"
            st.markdown(f'<div class="abox {css}">{action}  {icon}</div>', unsafe_allow_html=True)

            conf_icon = {"High": "🟢", "Medium": "🟡", "Close call": "🔴"}[confidence]
            st.caption(f"{conf_icon} **{confidence}** confidence")

            for tip in _tips(pc, total, dv):
                st.info(tip)

    else:
        missing = []
        if not pc:
            missing.append("your cards")
        if not dv:
            missing.append("dealer's card")
        st.markdown(
            f"<p style='color:#aaa; text-align:center; font-size:1.1em; margin:1rem 0'>"
            f"Add {' and '.join(missing)} to get a recommendation</p>",
            unsafe_allow_html=True,
        )

    # ── CARD COUNT (always visible, compact) ────────────────────────────────────
    all_cards = st.session_state.cards_seen + pc + ([dv] if dv else [])
    count = _hi_lo(all_cards)
    sign = "+" if count > 0 else ""
    if count > 2:
        count_color, count_msg = "#2dc653", "Deck favors the player"
    elif count < -2:
        count_color, count_msg = "#e63946", "Deck favors the dealer"
    else:
        count_color, count_msg = "#888", "Neutral shoe"
    st.markdown(
        f"<div style='text-align:center; margin:0.6rem 0; font-size:0.9em; color:{count_color}'>"
        f"  Hi-Lo count: <strong>{sign}{count}</strong>  ·  {count_msg}"
        f"</div>",
        unsafe_allow_html=True,
    )

    st.divider()

    # ── OUTCOME — one tap records and starts next hand ─────────────────────────
    st.markdown('<p class="section-label">What happened?</p>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)

    # Green WIN button
    st.markdown('<div data-outcome="win">', unsafe_allow_html=True)
    if c1.button("✅  WIN", use_container_width=True, key="btn_win"):
        _record_and_reset("win")
    st.markdown('</div>', unsafe_allow_html=True)

    # Red LOSS button
    st.markdown('<div data-outcome="loss">', unsafe_allow_html=True)
    if c2.button("❌  LOSS", use_container_width=True, key="btn_loss"):
        _record_and_reset("loss")
    st.markdown('</div>', unsafe_allow_html=True)

    # Gray DRAW button
    st.markdown('<div data-outcome="draw">', unsafe_allow_html=True)
    if c3.button("🤝  DRAW", use_container_width=True, key="btn_draw"):
        _record_and_reset("draw")
    st.markdown('</div>', unsafe_allow_html=True)

    # New hand without recording
    if st.button("🔄  New hand  (don't record)", use_container_width=True):
        st.session_state.player_cards = []
        st.session_state.dealer_value = None
        st.rerun()

    # ── SESSION STATS ──────────────────────────────────────────────────────────
    w = st.session_state.s_wins
    l = st.session_state.s_losses
    d = st.session_state.s_draws
    total = w + l + d

    if total > 0:
        st.divider()
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Hands", total)
        c2.metric("Wins", w, f"{w / total * 100:.0f}%")
        c3.metric("Losses", l)
        c4.metric("Draws", d)
        net_units = w - l
        sign2 = "+" if net_units >= 0 else ""
        st.caption(f"Net: **{sign2}{net_units}** units  ·  Bet sizing hint: {'bet more 💰' if count > 3 else 'bet minimum 🛑' if count < -3 else 'normal bet'}")
        if st.button("Reset session", key="reset"):
            for k in ("s_wins", "s_losses", "s_draws", "cards_seen"):
                st.session_state[k] = [] if k == "cards_seen" else 0
            st.rerun()


if __name__ == "__main__":
    main()
