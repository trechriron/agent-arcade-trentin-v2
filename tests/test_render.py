import os
# Comment out dummy driver and use proper one for macOS
# os.environ['SDL_VIDEODRIVER'] = 'dummy'  # Use a dummy video driver as fallback
os.environ['SDL_VIDEODRIVER'] = 'cocoa'  # Use macOS native driver

from cli.games import get_game
import time

print("SDL_VIDEODRIVER:", os.environ.get('SDL_VIDEODRIVER', 'not set'))
print("DISPLAY:", os.environ.get('DISPLAY', 'not set'))

# Create environment with rendering enabled
env = get_game('pong').make_env(render=True)
obs, info = env.reset()
print("Environment initialized successfully!")

# Loop through some random actions
for i in range(100):
    action = env.action_space.sample()
    # Handle both old and new gym step API
    step_result = env.step(action)
    if len(step_result) == 5:
        # New gym API: obs, reward, terminated, truncated, info
        obs, reward, terminated, truncated, info = step_result
        done = terminated or truncated
    else:
        # Old gym API: obs, reward, done, info
        obs, reward, done, info = step_result
    
    print(f"Step {i}, reward: {reward}")
    env.render()
    time.sleep(0.05)  # Slow down to see rendering
    
    if done:
        print("Episode finished, resetting...")
        obs, info = env.reset()

env.close()
print("Test completed successfully!") 