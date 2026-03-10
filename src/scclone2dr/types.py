from enum import Enum
# ---------------------------------------------------------------------------
# Mode enumerations
# ---------------------------------------------------------------------------

class NuMode(str, Enum):
    """Pre-assay (nu) effect parameterisation."""
    FIXED            = "fixed"
    NOISE_CORRECTION = "noise_correction"


class ThetaMode(str, Enum):
    """Overdispersion parameterisation."""
    NO_OVERDISPERSION    = "no_overdispersion"
    EQUAL                = "equal"
    SHARED               = "shared"
    NOT_SHARED_COUPLED   = "not shared coupled"
    NOT_SHARED_DECOUPLED = "not shared decoupled"