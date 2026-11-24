# Gel Performance Tool

> **Portfolio project built on synthetic data only.** No real process data, thresholds or company code are used.
> All numbers in this repository are invented to demonstrate a method.

For every completed chromatography cycle, the tool computes a standardised feature set, combines it into a
weighted performance score (0–100, higher = healthier gel) and recommends **OK / WATCH / REPLACE SOON**.

*(This README grows with each build step.)*

## Setup

```bash
cd gel-performance-tool
pip install -e ".[dashboard,dev]"
```
