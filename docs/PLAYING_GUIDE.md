# Blackjack AI Advisor — Playing Guide

Use the AI to get real-time action recommendations during a live Blackjack game.

---

## Quick Start (3 steps)

```bash
# 1. Make sure the model is trained
make train-dueling   # ~30 min on CPU — only needed once

# 2. Launch the advisor
make app             # opens http://localhost:8501 in your browser

# 3. Open it on your phone
# Find your local IP: hostname -I | awk '{print $1}'
# Then visit http://<your-ip>:8501 from your phone
```

---

## How to Use It During a Game

### Each hand, follow these steps:

1. **Tap your cards** — click each card as you receive it (2 clicks for a 7+7 pair, etc.)
2. **Select the dealer's visible card** — use the dropdown
3. **Read the recommendation** — the big colored box tells you what to do
4. **Act on it** — then click "New hand" or record the result

That's it. You don't need to manually enter your hand total — the AI computes it from your cards.

### What the colors mean

| Color | Action | Meaning |
|-------|--------|---------|
| 🔴 Red | **HIT** | Draw another card |
| 🟢 Green | **STAND** | Keep your hand as-is |
| ⭐ Gold | **BLACKJACK** | You have 21 with 2 cards — you win! |
| ⚫ Gray | **BUST** | You exceeded 21 — you lose this hand |

### Confidence indicator

- 🟢 **High** — the AI is very clear on what to do (margin > 0.25 Q-value units)
- 🟡 **Medium** — the AI leans one way but it's less decisive
- 🔴 **Close call** — genuinely borderline; either play could be reasonable

---

## Moves the AI Doesn't Control (use these tips instead)

The model only recommends Hit or Stand. For these special moves, follow the universal rules below — they are mathematically verified and not affected by card counting.

### Double Down
Double your bet and take exactly one more card. Always do this when:
- You have **11** — always
- You have **10** vs dealer **2–9**
- You have **9** vs dealer **3–6**
- You have a **soft 13–18** (Ace + something) vs dealer **4–6**

Never double on 12+.

### Split (when you have a pair)
| Your Pair | What to do |
|-----------|-----------|
| **Aces** | Always split — always |
| **8s** | Always split — 16 is the worst hand |
| **10s** | Never split — 20 is excellent |
| **5s** | Never split — treat as 10 and hit/double |
| **4s** | Never split (unless dealer shows 5 or 6) |
| **2s, 3s, 6s, 7s** | Split vs dealer 2–7, otherwise hit |
| **9s** | Split vs dealer 2–6 and 8–9; stand vs 7, 10, Ace |

> The app shows a tip in the recommendation area when you have one of the "always split" pairs.

### Surrender (if the casino allows it)
Give up half your bet and end the hand. Worth it when:
- Hard **16** vs dealer **9, 10, or Ace**
- Hard **15** vs dealer **10**

> The app shows a surrender tip when these exact situations arise.

---

## Card Count (Hi-Lo) — Bonus Feature

Open the "📊 Running Card Count" section to see the current Hi-Lo count.

**How it works:**
- Cards 2–6 dealt = **+1** (fewer low cards left → deck favors player)
- Cards 7–9 dealt = **0** (neutral)
- Cards 10/J/Q/K/A dealt = **–1** (fewer high cards left → deck favors dealer)

**What to do with it:**
| Count | What it means | Bet suggestion |
|-------|---------------|----------------|
| +3 or higher | Many high cards left — you have an edge | Bet more |
| –1 to +2 | Neutral deck | Normal bet |
| –3 or lower | Many low cards left — dealer has an edge | Bet your minimum |

The count resets when you click "Reset session stats". It persists across hands as long as you record results (which tells the app what cards were dealt).

> **Important:** The card count only matters in a multi-deck shoe game if you're tracking all cards. If your friends reshuffle every hand, counting is useless.

---

## Understanding the AI's Limitations

**The AI recommends the mathematically best play**, but Blackjack is still a game of chance. Here's what to expect:

| What to expect | Reality |
|----------------|---------|
| Win rate | ~43% of hands (even with perfect play) |
| House edge | ~0.5% — you'll lose about $0.50 per $100 bet long-term |
| When you'll win | ~43% wins, ~9% draws, ~48% losses |

This is the mathematical reality for any player — human or AI — using Hit/Stand only. The AI is helping you **make the best possible decision given the cards**, not guaranteeing wins.

The main advantage of following AI advice:
- Eliminates costly mistakes (like hitting on 20 or standing on 12 vs dealer 7)
- Reduces the house edge from ~3–4% (average player) to ~0.5% (near-optimal)
- That's worth $2.50–$3.50 saved per $100 bet compared to playing by gut

---

## Tips for Using This in a Real Game

1. **Keep your phone under the table or in your pocket** — use it discreetly
2. **Enter cards fast** — you don't need to record every card, just your hand and dealer's card
3. **"Record & next hand" tracks the count** — tap it after each hand even if you didn't win
4. **Reset count at each shoe** — when the dealer shuffles, reset session stats so the count restarts

---

## Troubleshooting

**"No trained model found" error**
```bash
make train-dueling   # trains and saves the model
```

**App not loading on phone**
```bash
# Find your computer's IP address
hostname -I | awk '{print $1}'
# Then open http://<that-ip>:8501 on your phone
# Make sure both devices are on the same WiFi
```

**App is slow**
The model runs on CPU and is very fast (< 1 ms per recommendation). If it's slow, it's likely a network/browser issue.
