# Agent Arcade Documentation

Welcome to the Agent Arcade documentation. This guide will help you train AI agents, participate in competitions, and earn rewards on the NEAR blockchain.

## Documentation Structure

### Core Guides

1. [Getting Started](getting-started.md)
   - Installation
   - Basic setup
   - Quick start guide
   - First agent training

2. [CLI Reference](cli-reference.md)
   - Complete command reference
   - Workflow examples
   - Best practices
   - Error handling

3. [Training Guide](training-guide.md)
   - Training configuration
   - Performance metrics
   - Optimization tips
   - Monitoring and visualization

4. [Competition Guide](competition-guide.md)
   - NEAR blockchain integration
   - Staking mechanism
   - Competition rules
   - Reward system

5. [Deployment Guide](deployment-guide.md)
   - Contract deployment
   - Game configuration
   - Security features
   - Monitoring and maintenance

### Important Note

Before participating in competitions:

1. **Login to NEAR wallet**:

   ```bash
   agent-arcade wallet-cmd login
   ```

2. **Evaluation & Score Submission Workflow**:

   ```bash
   # Evaluate to generate verification token
   agent-arcade evaluate pong [model_path] --episodes 10
   
   # Submit verified score
   agent-arcade stake submit pong 18
   ```

See [CLI Reference](cli-reference.md) for detailed command options.

### Advanced Topics

[Adding New Games](adding-games.md)

- Game integration guide
- Environment setup
- Testing requirements
- Submission process

## Quick Links

- [GitHub Repository](https://github.com/jbarnes850/agent-arcade)
- [NEAR Protocol](https://near.org)
- [Issue Tracker](https://github.com/jbarnes850/agent-arcade/issues)

## Getting Help

1. **Documentation Issues**
   - Use the [issue tracker](https://github.com/jbarnes850/agent-arcade/issues)
   - Tag issues with `documentation`

2. **Technical Support**
   - Join our [Discord community](https://discord.gg/near)
   - Tag: `#agent-arcade`

3. **Contributing**
   - See [Adding New Games](adding-games.md)

## Documentation Updates

This documentation is regularly updated. Major sections:

| Section | Last Updated | Primary Focus |
|---------|--------------|---------------|
| Getting Started | Feb 24, 2025 | Installation, basic usage |
| CLI Reference | Feb 24, 2025 | Command documentation |
| Training Guide | Feb 24, 2025 | Configuration, metrics |
| Competition Guide | Feb 24, 2025 | NEAR integration |
| Deployment Guide | Feb 24, 2025 | Contract deployment |
| Adding Games | Feb 24, 2025 | Game development |
