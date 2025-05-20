from types import ModuleType, SimpleNamespace

try:
    import gymnasium as gym
    from gymnasium import spaces

    Space = spaces.Space

except ImportError:
    gym: ModuleType = SimpleNamespace(spaces= SimpleNamespace(Space=SimpleNamespace(),Dict=SimpleNamespace()))
    spaces = gym.spaces
    gymnasium: ModuleType = ModuleType("gymnasium")
