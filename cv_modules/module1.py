import numpy as np
from typing import Dict, Any

# Camera intrinsics are fixed properties of the calibrated camera.
FX = 3081.84216
FY = 3077.33506

def calculate_dimension_m1(data: Dict[str, Any]) -> Dict[str, float]:
    """
    Computes real-world dimensions (dX, dY) and error from pixel coordinates 
    (p1, p2) and known object depth (Z) using perspective projection.
    
    Formula: dX = (dx * Z) / FX, dY = (dy * Z) / FY
    """
    try:
        p1 = data['p1']
        p2 = data['p2']
        Z = float(data['Z'])

        # Pixel differences
        dx = abs(p2['x'] - p1['x'])
        dy = abs(p2['y'] - p1['y'])

        # Real-world dimensions 
        dX = (dx * Z) / FX
        dY = (dy * Z) / FY

        # Real-world dimensions for error check (cm)
        real_dX = float(data.get('real_dX', 0))
        real_dY = float(data.get('real_dY', 0))

        error = 0.0
        
        # Calculate error
        if real_dX > 0 and real_dY > 0:
            error_X = abs(real_dX - dX)
            error_Y = abs(real_dY - dY)
            error = min(error_X, error_Y)
        elif real_dX > 0:
             error = abs(real_dX - dX)
        elif real_dY > 0:
             error = abs(real_dY - dY)

        return {
            'p1': p1,
            'p2': p2,
            'dx': round(dx, 4),
            'dy': round(dy, 4),
            'dX': round(dX, 4),
            'dY': round(dY, 4),
            'error': round(error, 4)
        }
    except Exception as e:
        raise ValueError(f"Module 1 Calculation failed: {str(e)}")