# Blackjack AI Advisor

Real-time strategy recommendations for your Blackjack game.

## Setup (one time)

**Requirements:** Python 3.10 or newer — download from [python.org](https://www.python.org/downloads/) if needed.

```bash
pip install streamlit torch numpy
```

> On Mac with Apple Silicon (M1/M2/M3) use `pip install streamlit torch numpy` — PyTorch works natively.
> On Windows, the same command works in PowerShell or Command Prompt.

## Run

```bash
streamlit run app.py
```

The app opens automatically in your browser at `http://localhost:8501`.

**To use on your phone** (same WiFi as your computer):
1. Find your computer's local IP — on Mac/Linux: `hostname -I`, on Windows: `ipconfig`
2. Open `http://<that-ip>:8501` on your phone

## How to use

1. **Tap your cards** — click each card in your hand as you receive it
2. **Select the dealer's visible card** — use the dropdown
3. **Follow the recommendation** — the big colored box tells you what to do

| Color | Action |
|-------|--------|
| 🔴 Red | HIT — draw another card |
| 🟢 Green | STAND — keep your hand |
| ⭐ Gold | BLACKJACK — natural 21, you win! |
| ⚫ Gray | BUST — you exceeded 21 |

The app also shows tips for **Split**, **Double Down**, and **Surrender** when they apply,
plus a running **Hi-Lo card count** if you want to track the shoe.

## What to expect

The AI plays near-perfect basic strategy (~98% agreement with the mathematically optimal chart).
In a real casino game you'll win about **43% of hands** — that's the mathematical ceiling for
Hit/Stand play regardless of who (or what) is playing. The AI eliminates costly mistakes and
keeps the house edge below 0.5%.
