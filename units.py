# units.py — Conversor de unidades simple, extensible y sin dependencias
# Categorías soportadas:
# - pressure: Pa, kPa, MPa, bar, psi, atm, kgf/cm2
# - flow: m3/s, m3/d, stb/d, bbl/d
# - volume: m3, bbl (stb)
# - length: m, cm, mm, km, in, ft, yd, mile
# - temperature (afín): K, C, F, R
# - viscosity: Pa·s, cP
# - permeability: m2, darcy, md, nd
#
# API principal:
#   list_units(category) -> lista de strings
#   convert(category, value, from_unit, to_unit) -> float
#   add_unit_linear(category, unit, to_SI_factor)
#   add_unit_affine(category, unit, to_SI_fn, from_SI_fn)

from __future__ import annotations
from typing import Callable, Dict, Tuple

# ----------------------------
# Definición interna
# ----------------------------

# Categorías lineales: conversión por factor multiplicativo a SI
_LINEAR_FACTORS: Dict[str, Dict[str, float]] = {
    "pressure": {
        "Pa": 1.0,
        "kPa": 1e3,
        "MPa": 1e6,
        "bar": 1e5,
        "psi": 6894.757293168,
        "atm": 101325.0,
        "kgf/cm2": 98066.5,   # kgf por cm^2 ≈ 98,066.5 Pa
    },
    "flow": {
        "m3/s": 1.0,
        "m3/d": 1.0 / 86400.0,
        "stb/d": 0.158987294928 / 86400.0,
        "bbl/d": 0.158987294928 / 86400.0,  # alias
    },
    "volume": {
        "m3": 1.0,
        "stb": 0.158987294928,
        "bbl": 0.158987294928,  # alias
    },
    "length": {
        "m": 1.0,
        "cm": 0.01,
        "mm": 0.001,
        "km": 1000.0,
        "in": 0.0254,
        "ft": 0.3048,
        "yd": 0.9144,
        "mile": 1609.344,
    },
    "viscosity": {
        "Pa·s": 1.0,
        "cP": 1e-3,  # 1 cP = 1 mPa·s = 1e-3 Pa·s
    },
    "permeability": {
        "m2": 1.0,
        "darcy": 9.869233e-13,
        "md": 9.869233e-16,   # milli-darcy
        "nd": 9.869233e-22,   # nano-darcy (útil en no convencionales)
    },
}

# Categorías afines: requieren offset + escala (p.ej. temperatura)
# Cada unidad define par de funciones: (to_SI, from_SI), donde SI es Kelvin
_AFFINE_UNITS: Dict[str, Dict[str, Tuple[Callable[[float], float], Callable[[float], float]]]] = {
    "temperature": {
        "K":  (lambda x: x,                lambda k: k),
        "C":  (lambda c: c + 273.15,       lambda k: k - 273.15),
        "F":  (lambda f: (f + 459.67) * 5/9, lambda k: k * 9/5 - 459.67),
        "R":  (lambda r: r * 5/9,          lambda k: k * 9/5),
    }
}

# Nombre de unidad SI por categoría (para referencia interna)
_SI_UNIT: Dict[str, str] = {
    "pressure": "Pa",
    "flow": "m3/s",
    "volume": "m3",
    "length": "m",
    "temperature": "K",
    "viscosity": "Pa·s",
    "permeability": "m2",
}

# ----------------------------
# API pública
# ----------------------------

def list_units(category: str) -> list[str]:
    """Lista las unidades disponibles para una categoría."""
    if category in _LINEAR_FACTORS:
        return sorted(_LINEAR_FACTORS[category].keys())
    if category in _AFFINE_UNITS:
        return sorted(_AFFINE_UNITS[category].keys())
    return []

def convert(category: str, value: float, from_unit: str, to_unit: str) -> float:
    """
    Convierte 'value' de 'from_unit' a 'to_unit' dentro de la misma categoría.
    - Lineales: value * factor(from) / factor(to)
    - Afines (temperatura): aplica transformación con offset.
    """
    if category in _LINEAR_FACTORS:
        factors = _LINEAR_FACTORS[category]
        if from_unit not in factors:
            raise ValueError(_err_units(category, from_unit, factors.keys()))
        if to_unit not in factors:
            raise ValueError(_err_units(category, to_unit, factors.keys()))
        to_SI = factors[from_unit]
        from_SI = 1.0 / factors[to_unit]
        return float(value) * to_SI * from_SI

    if category in _AFFINE_UNITS:
        units = _AFFINE_UNITS[category]
        if from_unit not in units:
            raise ValueError(_err_units(category, from_unit, units.keys()))
        if to_unit not in units:
            raise ValueError(_err_units(category, to_unit, units.keys()))
        to_SI_fn, _ = units[from_unit]
        _, from_SI_fn = units[to_unit]
        return float(from_SI_fn(to_SI_fn(value)))

    raise ValueError(f"Categoría desconocida: {category}")

def add_unit_linear(category: str, unit: str, to_SI_factor: float) -> None:
    """
    Agrega/actualiza una unidad lineal en la categoría dada.
    Ej.: add_unit_linear('flow', 'm3/h', 1.0/3600.0)
    """
    if category not in _LINEAR_FACTORS:
        _LINEAR_FACTORS[category] = {}
    _LINEAR_FACTORS[category][unit] = float(to_SI_factor)

def add_unit_affine(category: str, unit: str,
                    to_SI_fn: Callable[[float], float],
                    from_SI_fn: Callable[[float], float]) -> None:
    """
    Agrega/actualiza una unidad afín (con offset), p.ej. temperatura.
    Ej.: add_unit_affine('temperature', 'De', to_SI_fn, from_SI_fn)
    """
    if category not in _AFFINE_UNITS:
        _AFFINE_UNITS[category] = {}
    _AFFINE_UNITS[category][unit] = (to_SI_fn, from_SI_fn)

# Helpers específicos por categoría (azúcar sintáctica)
def conv_pressure(val: float, src: str, dst: str) -> float:
    return convert("pressure", val, src, dst)

def conv_flow(val: float, src: str, dst: str) -> float:
    return convert("flow", val, src, dst)

def conv_volume(val: float, src: str, dst: str) -> float:
    return convert("volume", val, src, dst)

def conv_length(val: float, src: str, dst: str) -> float:
    return convert("length", val, src, dst)

def conv_temperature(val: float, src: str, dst: str) -> float:
    return convert("temperature", val, src, dst)

def conv_viscosity(val: float, src: str, dst: str) -> float:
    return convert("viscosity", val, src, dst)

def conv_perm(val: float, src: str, dst: str) -> float:
    return convert("permeability", val, src, dst)

# ----------------------------
# Utilidades
# ----------------------------

def si_unit(category: str) -> str:
    """Devuelve el símbolo de la unidad SI de la categoría."""
    return _SI_UNIT.get(category, "?")

def _err_units(category: str, unit: str, choices) -> str:
    return (f"Unidad desconocida '{unit}' para categoría '{category}'. "
            f"Opciones: {sorted(list(choices))}")

# ----------------------------
# Ejemplos de uso (comentados)
# ----------------------------
# p_psi = conv_pressure(30.0, "MPa", "psi")       # ~ 435113.78 psi
# q_m3s = conv_flow(800, "stb/d", "m3/s")         # ~ 0.001474 m3/s
# L_ft  = conv_length(120.0, "m", "ft")           # ~ 393.701 ft
# T_F   = conv_temperature(25.0, "C", "F")        # 77.0 F
# mu_Pa = conv_viscosity(0.7, "cP", "Pa·s")       # 0.0007 Pa·s
# k_md  = conv_perm(100.0, "md", "darcy")         # 0.1 darcy
# add_unit_linear("flow", "m3/h", 1.0/3600.0)     # agregar m3/h si lo necesitás
