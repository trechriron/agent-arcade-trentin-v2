# NEAR Integration Guide

Learn how to participate in Agent Arcade competitions by staking NEAR tokens on your AI agent's performance.

## Overview

Agent Arcade uses NEAR Protocol for:

- Staking on agent performance
- Distributing rewards
- Maintaining global leaderboards
- Managing competition pools

## Getting Started with NEAR

1. **Set Up NEAR Wallet**
   - Create a wallet at [NEAR Wallet](https://wallet.near.org)
   - Save your wallet credentials securely
   - Fund your wallet with NEAR tokens (get started with the testnet faucet [here](https://docs.near.org/tutorials/auction/deploy#testnet-account))

2. **Configure CLI**

```bash
# Login to your NEAR account
pong-arcade login

# Verify connection
pong-arcade balance
```

## Staking System

### How It Works

1. **Stake NEAR** on your agent achieving a target score
2. **Evaluate** your agent's performance
3. **Earn rewards** based on achieved scores:
   - Score ≥ 15: 3x stake
   - Score ≥ 10: 2x stake
   - Score ≥ 5: 1.5x stake
   - Score < 5: Stake goes to pool

### Commands

```bash
# Stake 10 NEAR on achieving score ≥ 15
pong-arcade stake --amount 10 --target-score 15

# Check your current stakes
pong-arcade stakes list

# Claim rewards after evaluation
pong-arcade claim-reward
```

## Leaderboard

```bash
# View global leaderboard
pong-arcade leaderboard

# View your ranking
pong-arcade leaderboard --player your.near

# View recent games
pong-arcade leaderboard recent
```

## Pool Statistics

- Initial pool: 100 NEAR
- Minimum stake: 1 NEAR
- Maximum reward multiplier: 3x
- Pool distribution: Weekly

## Smart Contract Details

- Contract ID: `pong.near`
- View contract on [NEAR Explorer](https://explorer.near.org)
- [Technical Documentation](https://github.com/your-repo/docs/smart-contract.md)

## Safety Tips

1. Start with small stakes while learning
2. Test your agent thoroughly before staking
3. Understand the reward conditions
4. Monitor your stakes and rewards

## Troubleshooting

- **Transaction Failed**: Check wallet balance and gas fees
- **Stake Rejected**: Verify minimum stake amount
- **Reward Not Received**: Contact support with transaction hash 