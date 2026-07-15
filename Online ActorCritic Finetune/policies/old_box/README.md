# Old-box policies (pre-tuning archive)

The four ORIGINAL best policies of the online actor-critic fine-tuning, trained
in the **previous (narrow) gain box** and archived here before the tuned
policies replaced them in `../`:

```
GAIN_LOW  = [-6.0, -0.100, -0.60]     # [Kp, Ki, Kw]
GAIN_HIGH = [-0.05, -0.0001, 0.10]
```

**IMPORTANT — decoding box:** these actors output normalised gains in [0,1]^3
that must be decoded through the box ABOVE (the default in `config.py`). Do NOT
load them with the tuned box (`GAIN_LOW=[-30,-4.0,-0.4]`) that the current
`*_best.pt` policies in `../` use — the physical gains would be wrong.

Results on Juan's 4 days (full-day, mean MAE / worst overshoot / worst
undershoot):

| policy | MAE (C) | overshoot (C) | undershoot (C) |
|---|---|---|---|
| ddpg_online_ac_approach1 | 0.109 | 3.64 | 1.10 |
| ddpg_online_ac_approach2 | 0.109 | 3.64 | 1.09 |
| td3_online_ac_approach1  | 0.109 | 3.64 | 1.09 |
| td3_online_ac_approach2  | 0.183 | 4.03 | 2.22 |

The matching notebook code + outputs for these policies are preserved in git
history (last old-box commit of the three approach1/2 policies: parent of
`2c956f6`; td3_approach2's old run: commit `2c956f6` itself). The tuned
replacements (stability-safe wide box, Kw*Ki*TS <= 1.6, undershoot-aware
selection) live in `../` and are documented in the four approach notebooks.
