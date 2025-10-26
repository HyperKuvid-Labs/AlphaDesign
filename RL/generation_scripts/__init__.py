"""
F1 Front Wing Specialized Generators
Modular generation system for ultra-realistic F1 wings
"""

__version__ = "1.0.0"

# Optional: Export classes for easier imports
try:
    from .f1_multi_flap_system_gen import F1FrontWingMultiElementGenerator
    from .f1_main_wing_geometry import F1FrontWingMainElementGenerator
    from .f1_y250_gen import F1FrontWingY250CentralStructureGenerator
    from .f1_endplate_generator import F1FrontWingEndplateGenerator

    __all__ = [
        'F1FrontWingMultiElementGenerator',
        'F1FrontWingMainElementGenerator',
        'F1FrontWingY250CentralStructureGenerator',
        'F1FrontWingEndplateGenerator'
    ]
except ImportError as e:
    print(f"Warning: Some specialized generators could not be imported: {e}")
    __all__ = []