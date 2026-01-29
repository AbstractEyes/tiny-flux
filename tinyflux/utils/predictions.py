"""
Prediction type conversions for diffusion models.

Flow Matching:  x_t = (1-t)*noise + t*data,  v = data - noise
V-Prediction:   x_t = sqrt(α)*data + sqrt(1-α)*noise,  v = sqrt(α)*noise - sqrt(1-α)*data
Epsilon:        x_t = sqrt(α)*data + sqrt(1-α)*noise,  ε = noise
"""

# =============================================================================
# Flow Matching (Rectified Flow)
# =============================================================================

def flow_x_t(data, noise, t):
    """Interpolate: x_t = (1-t)*noise + t*data"""
    t = t.view(-1, *([1] * (data.ndim - 1)))
    return (1 - t) * noise + t * data


def flow_velocity(data, noise):
    """Target: v = data - noise"""
    return data - noise


def flow_data_from_velocity(x_t, v, t):
    """Recover: data = x_t + (1-t)*v"""
    t = t.view(-1, *([1] * (x_t.ndim - 1)))
    return x_t + (1 - t) * v


def flow_noise_from_velocity(x_t, v, t):
    """Recover: noise = x_t - t*v"""
    t = t.view(-1, *([1] * (x_t.ndim - 1)))
    return x_t - t * v


# =============================================================================
# V-Prediction
# =============================================================================

def vpred_x_t(data, noise, alpha):
    """Interpolate: x_t = sqrt(α)*data + sqrt(1-α)*noise"""
    sqrt_a = alpha.sqrt().view(-1, *([1] * (data.ndim - 1)))
    sqrt_1ma = (1 - alpha).sqrt().view(-1, *([1] * (data.ndim - 1)))
    return sqrt_a * data + sqrt_1ma * noise


def vpred_velocity(data, noise, alpha):
    """Target: v = sqrt(α)*noise - sqrt(1-α)*data"""
    sqrt_a = alpha.sqrt().view(-1, *([1] * (data.ndim - 1)))
    sqrt_1ma = (1 - alpha).sqrt().view(-1, *([1] * (data.ndim - 1)))
    return sqrt_a * noise - sqrt_1ma * data


def vpred_data_from_velocity(x_t, v, alpha):
    """Recover: data = sqrt(α)*x_t - sqrt(1-α)*v"""
    sqrt_a = alpha.sqrt().view(-1, *([1] * (x_t.ndim - 1)))
    sqrt_1ma = (1 - alpha).sqrt().view(-1, *([1] * (x_t.ndim - 1)))
    return sqrt_a * x_t - sqrt_1ma * v


def vpred_noise_from_velocity(x_t, v, alpha):
    """Recover: noise = sqrt(1-α)*x_t + sqrt(α)*v"""
    sqrt_a = alpha.sqrt().view(-1, *([1] * (x_t.ndim - 1)))
    sqrt_1ma = (1 - alpha).sqrt().view(-1, *([1] * (x_t.ndim - 1)))
    return sqrt_1ma * x_t + sqrt_a * v


# =============================================================================
# Epsilon Prediction
# =============================================================================

def eps_x_t(data, noise, alpha):
    """Interpolate: x_t = sqrt(α)*data + sqrt(1-α)*noise"""
    sqrt_a = alpha.sqrt().view(-1, *([1] * (data.ndim - 1)))
    sqrt_1ma = (1 - alpha).sqrt().view(-1, *([1] * (data.ndim - 1)))
    return sqrt_a * data + sqrt_1ma * noise


def eps_target(noise):
    """Target: ε = noise"""
    return noise


def eps_data_from_noise(x_t, eps, alpha):
    """Recover: data = (x_t - sqrt(1-α)*ε) / sqrt(α)"""
    sqrt_a = alpha.sqrt().view(-1, *([1] * (x_t.ndim - 1)))
    sqrt_1ma = (1 - alpha).sqrt().view(-1, *([1] * (x_t.ndim - 1)))
    return (x_t - sqrt_1ma * eps) / sqrt_a


# =============================================================================
# Conversions
# =============================================================================

def eps_to_vpred(eps, x_t, alpha):
    """Convert ε → v"""
    sqrt_a = alpha.sqrt().view(-1, *([1] * (x_t.ndim - 1)))
    sqrt_1ma = (1 - alpha).sqrt().view(-1, *([1] * (x_t.ndim - 1)))
    data = (x_t - sqrt_1ma * eps) / sqrt_a
    return sqrt_a * eps - sqrt_1ma * data


def vpred_to_eps(v, x_t, alpha):
    """Convert v → ε"""
    sqrt_a = alpha.sqrt().view(-1, *([1] * (x_t.ndim - 1)))
    sqrt_1ma = (1 - alpha).sqrt().view(-1, *([1] * (x_t.ndim - 1)))
    return sqrt_1ma * x_t + sqrt_a * v