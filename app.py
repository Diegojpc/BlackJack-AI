"""
Blackjack AI Advisor — real-time strategy assistant.

Run with:
    streamlit run app.py
or:
    make app
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import streamlit as st
import torch

sys.path.insert(0, str(Path(__file__).parent))

from src.agents.deep.dueling_dqn import DuelingDQNAgent
from src.utils.config import DQNConfig, MODELS_DIR


# ─── Card helpers ─────────────────────────────────────────────────────────────

CARD_LABELS = ["A", "2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K"]
LABEL_TO_VALUE = {
    "A": 1, "2": 2, "3": 3, "4": 4, "5": 5,
    "6": 6, "7": 7, "8": 8, "9": 9,
    "10": 10, "J": 10, "Q": 10, "K": 10,
}
VALUE_TO_LABEL = {v: k for k, v in LABEL_TO_VALUE.items() if k not in ("J", "Q", "K")}
VALUE_TO_LABEL[10] = "10"
VALUE_TO_LABEL[1] = "A"


def hand_value(cards: list[int]) -> tuple[int, bool]:
    """Return (total, usable_ace). Ace is 11 when it doesn't bust."""
    total = sum(cards)
    usable_ace = 1 in cards and total + 10 <= 21
    return total + (10 if usable_ace else 0), usable_ace


def cards_display(cards: list[int]) -> str:
    return "  +  ".join(VALUE_TO_LABEL.get(c, str(c)) for c in cards)


def hi_lo_count(cards: list[int]) -> int:
    """Hi-Lo running count: 2–6 = +1, 7–9 = 0, 10/A = –1."""
    total = 0
    for c in cards:
        if 2 <= c <= 6:
            total += 1
        elif c == 10 or c == 1:
            total -= 1
    return total


# ─── Model loading ────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading AI model…")
def load_agent() -> DuelingDQNAgent | None:
    """
    Load the best available Dueling DQN checkpoint.

    Reads the saved config from the checkpoint so the network architecture
    matches exactly, regardless of current DQNConfig defaults.
    """
    candidates = [
        MODELS_DIR / "dueling_dqn_final.pt",
        *sorted(MODELS_DIR.glob("dueling_dqn_step*.pt"), reverse=True),
        *sorted(MODELS_DIR.glob("double_dqn_step*.pt"), reverse=True),
        MODELS_DIR / "dqn_final.pt",
    ]
    for path in candidates:
        if path.exists():
            try:
                ckpt = torch.load(path, map_location="cpu", weights_only=False)
                config: DQNConfig = ckpt.get("config", DQNConfig())
                config.device = "cpu"
                agent = DuelingDQNAgent(config)
                agent.load(path)
                return agent
            except Exception:
                continue
    return None


# ─── Strategy engine ──────────────────────────────────────────────────────────

def get_recommendation(
    agent: DuelingDQNAgent,
    player_cards: list[int],
    dealer_value: int,
) -> tuple[str, str, list[str]]:
    """
    Return (action, confidence_label, tips).

    action: "HIT" | "STAND" | "BUST" | "BLACKJACK"
    confidence_label: "High" | "Medium" | "Close call" | ""
    tips: list of contextual advice strings
    """
    player_sum, usable_ace = hand_value(player_cards)

    if player_sum > 21:
        return "BUST", "", []
    if player_sum == 21 and len(player_cards) == 2:
        return "BLACKJACK", "High", ["Natural Blackjack! You win 1.5x unless dealer also has 21."]
    if player_sum == 21:
        return "STAND", "High", ["You have 21 — always stand."]

    state = (player_sum, dealer_value, usable_ace)
    action_int = agent.get_action(state, greedy=True)

    # Q-value margin → confidence
    obs = torch.FloatTensor([
        player_sum / 31.0,
        dealer_value / 10.0,
        float(usable_ace),
    ]).unsqueeze(0).to(agent.device)
    with torch.no_grad():
        q = agent.online_net(obs).cpu().numpy()[0]
    margin = abs(float(q[1] - q[0]))

    confidence = "High" if margin > 0.25 else "Medium" if margin > 0.08 else "Close call"
    action_label = "HIT" if action_int == 1 else "STAND"
    tips = _build_tips(player_cards, player_sum, dealer_value, usable_ace)
    return action_label, confidence, tips


def _build_tips(
    cards: list[int], player_sum: int, dealer: int, usable_ace: bool
) -> list[str]:
    tips = []
    if len(cards) == 2:
        c1, c2 = cards
        # Pair advice
        if c1 == c2:
            if c1 == 1:
                tips.append("✂️ **Always Split Aces** — each hand restarts with a powerful Ace.")
            elif c1 == 8:
                tips.append("✂️ **Always Split 8s** — a hard 16 is the worst hand; 8+8→two 8s is much better.")
            elif c1 == 10:
                tips.append("🚫 **Never Split 10s** — 20 is one of the strongest hands; don't give it up.")
            elif c1 in (4, 5):
                tips.append(f"🚫 **Don't Split {c1}s** — treat this as a {'8' if c1==4 else '10'} and hit/double.")
        # Double Down opportunities
        if player_sum == 11:
            tips.append("⬆️ **Consider Doubling Down** — 11 vs any dealer card is your best doubling spot.")
        elif player_sum == 10 and dealer <= 9:
            tips.append(f"⬆️ **Consider Doubling Down** — 10 vs dealer {VALUE_TO_LABEL.get(dealer, dealer)} is a strong double.")
        elif player_sum == 9 and 3 <= dealer <= 6:
            tips.append(f"⬆️ **Consider Doubling Down** — 9 vs dealer {VALUE_TO_LABEL.get(dealer, dealer)} (weak) is a marginal double.")
    # Surrender hint
    if player_sum == 16 and dealer in (9, 10, 1) and len(cards) == 2:
        tips.append("🏳️ **Consider Surrender** (if allowed) — hard 16 vs 9/10/A is the classic surrender spot.")
    if player_sum == 15 and dealer == 10 and len(cards) == 2:
        tips.append("🏳️ **Consider Surrender** (if allowed) — hard 15 vs dealer 10 saves you half your bet.")
    return tips


# ─── Session state init ───────────────────────────────────────────────────────

def init_state() -> None:
    defaults = {
        "player_cards": [],
        "dealer_value": None,
        "s_wins": 0,
        "s_losses": 0,
        "s_draws": 0,
        "cards_seen": [],  # across completed hands, for running count
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


# ─── App ──────────────────────────────────────────────────────────────────────

def main() -> None:
    st.set_page_config(
        page_title="Blackjack AI Advisor",
        page_icon="🃏",
        layout="centered",
        initial_sidebar_state="collapsed",
    )

    st.markdown("""
    <style>
    /* Large card buttons */
    div[data-testid="column"] > div > div > div > button {
        font-size: 1.15em !important;
        font-weight: bold !important;
        padding: 0.4rem 0.1rem !important;
    }
    /* Action box */
    .action-box {
        text-align: center;
        padding: 1.2rem;
        border-radius: 14px;
        font-size: 2.6em;
        font-weight: 900;
        letter-spacing: 2px;
        margin: 0.6rem 0;
    }
    .action-hit        { background: #ff4b4b; color: #fff; }
    .action-stand      { background: #21c354; color: #fff; }
    .action-bust       { background: #808495; color: #fff; }
    .action-blackjack  { background: #ffd700; color: #1a1a1a; }
    </style>
    """, unsafe_allow_html=True)

    init_state()

    st.title("🃏 Blackjack AI Advisor")
    st.caption("Powered by a Dueling DQN trained to 98% basic-strategy accuracy")

    # ── Load model ─────────────────────────────────────────────────────────────
    agent = load_agent()
    if agent is None:
        st.error(
            "No trained model found in `models/`. "
            "Run `make train-dueling` to train one first (takes ~30 min on CPU)."
        )
        st.stop()

    # ── YOUR HAND ──────────────────────────────────────────────────────────────
    st.subheader("Your Hand")
    st.caption("Tap a card each time you receive one")

    cols = st.columns(len(CARD_LABELS))
    for i, label in enumerate(CARD_LABELS):
        if cols[i].button(label, key=f"p_{label}"):
            st.session_state.player_cards.append(LABEL_TO_VALUE[label])
            st.rerun()

    if st.session_state.player_cards:
        total, soft = hand_value(st.session_state.player_cards)
        hand_type = "Soft" if soft else "Hard"
        st.markdown(
            f"### {total} ({hand_type})"
            f"<span style='font-size:0.9em; color:gray;'> — {cards_display(st.session_state.player_cards)}</span>",
            unsafe_allow_html=True,
        )
        if st.button("↩ Remove last card", key="undo"):
            st.session_state.player_cards.pop()
            st.rerun()
    else:
        st.markdown("*No cards yet — tap above to add*")

    # ── DEALER'S UP CARD ───────────────────────────────────────────────────────
    st.subheader("Dealer's Up Card")
    dealer_choice = st.selectbox(
        "Dealer showing",
        ["— select —"] + CARD_LABELS,
        label_visibility="collapsed",
        key="dealer_select",
    )
    if dealer_choice != "— select —":
        st.session_state.dealer_value = LABEL_TO_VALUE[dealer_choice]
    else:
        st.session_state.dealer_value = None

    # ── AI RECOMMENDATION ──────────────────────────────────────────────────────
    st.divider()

    player_ready = bool(st.session_state.player_cards)
    dealer_ready = st.session_state.dealer_value is not None

    if player_ready and dealer_ready:
        action, confidence, tips = get_recommendation(
            agent,
            st.session_state.player_cards,
            st.session_state.dealer_value,
        )

        emoji_map = {"HIT": "🎯", "STAND": "✋", "BUST": "💀", "BLACKJACK": "⭐"}
        css_map = {"HIT": "action-hit", "STAND": "action-stand",
                   "BUST": "action-bust", "BLACKJACK": "action-blackjack"}

        st.markdown(
            f'<div class="action-box {css_map[action]}">'
            f'{action} {emoji_map.get(action, "")}'
            f'</div>',
            unsafe_allow_html=True,
        )

        if confidence:
            icons = {"High": "🟢", "Medium": "🟡", "Close call": "🔴"}
            st.caption(f"{icons.get(confidence, '')} Confidence: **{confidence}**")

        for tip in tips:
            st.info(tip)

    elif player_ready:
        st.info("Select the dealer's card to get a recommendation")
    elif dealer_ready:
        st.info("Add your cards to get a recommendation")
    else:
        st.markdown("*Add your cards and the dealer's card above*")

    # ── HI-LO CARD COUNT ───────────────────────────────────────────────────────
    with st.expander("📊 Running Card Count (Hi-Lo)", expanded=False):
        visible = list(st.session_state.player_cards)
        if st.session_state.dealer_value:
            visible.append(st.session_state.dealer_value)
        count = hi_lo_count(st.session_state.cards_seen + visible)
        sign = "+" if count > 0 else ""

        if count > 2:
            st.success(f"Count: **{sign}{count}** — High count: more 10s/Aces left → player advantage 🎉")
        elif count < -2:
            st.warning(f"Count: **{count}** — Low count: more small cards left → dealer advantage ⚠️")
        else:
            st.info(f"Count: **{sign}{count}** — Roughly neutral shoe")

        st.caption("2–6 = **+1** &nbsp;&nbsp; 7–9 = **0** &nbsp;&nbsp; 10/J/Q/K/A = **−1**")

        if abs(count) > 3:
            bet_tip = "Bet more than your usual amount" if count > 3 else "Bet your minimum"
            st.caption(f"💰 Betting tip: {bet_tip} (count = {sign}{count})")

    # ── HAND CONTROLS ──────────────────────────────────────────────────────────
    st.divider()
    col_result, col_new = st.columns([3, 1])

    with col_result:
        result_options = ["Record outcome…", "Win ✅", "Loss ❌", "Draw / Push 🤝"]
        result = st.selectbox(
            "Outcome",
            result_options,
            label_visibility="collapsed",
            key="outcome_select",
        )
        if result != result_options[0]:
            if st.button("Save & next hand", type="primary"):
                # Archive cards for running count
                hand_cards = list(st.session_state.player_cards)
                if st.session_state.dealer_value:
                    hand_cards.append(st.session_state.dealer_value)
                st.session_state.cards_seen.extend(hand_cards)

                if "Win" in result:
                    st.session_state.s_wins += 1
                elif "Loss" in result:
                    st.session_state.s_losses += 1
                else:
                    st.session_state.s_draws += 1

                st.session_state.player_cards = []
                st.session_state.dealer_value = None
                st.rerun()

    with col_new:
        if st.button("🔄 New hand", use_container_width=True):
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
        st.subheader("Session Stats")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Hands", total)
        c2.metric("Wins", w, delta=f"{w / total * 100:.0f}%")
        c3.metric("Losses", l)
        c4.metric("Draws", d)
        net = w - l
        sign = "+" if net >= 0 else ""
        st.caption(f"Net (assuming $1/hand): **{sign}{net}** units")
        if st.button("Reset session stats"):
            st.session_state.s_wins = 0
            st.session_state.s_losses = 0
            st.session_state.s_draws = 0
            st.session_state.cards_seen = []
            st.rerun()


if __name__ == "__main__":
    main()
