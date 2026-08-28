import torch


def evaluate_sh(
    coefficients: torch.Tensor,
    directions: torch.Tensor,
    degree: int,
) -> torch.Tensor:
    """Evaluate real spherical harmonics through degree 3."""
    if degree < 0 or degree > 3:
        raise ValueError("the reference renderer supports SH degrees from 0 through 3")
    if coefficients.ndim != 3 or coefficients.shape[-1] != 3:
        raise ValueError("coefficients must have shape (N, coefficients, 3)")
    if directions.shape != (coefficients.shape[0], 3):
        raise ValueError("directions must have shape (N, 3)")
    required = (degree + 1) ** 2
    if coefficients.shape[1] < required:
        raise ValueError(f"SH degree {degree} requires at least {required} coefficients")

    x, y, z = directions.unbind(dim=-1)
    result = 0.28209479177387814 * coefficients[:, 0]
    if degree >= 1:
        result = (
            result
            - 0.4886025119029199 * y[:, None] * coefficients[:, 1]
            + 0.4886025119029199 * z[:, None] * coefficients[:, 2]
            - 0.4886025119029199 * x[:, None] * coefficients[:, 3]
        )
    if degree >= 2:
        xx, yy, zz = x.square(), y.square(), z.square()
        result = (
            result
            - 1.0925484305920792 * (x * y)[:, None] * coefficients[:, 4]
            + 1.0925484305920792 * (y * z)[:, None] * coefficients[:, 5]
            + 0.31539156525252005
            * (2.0 * zz - xx - yy)[:, None]
            * coefficients[:, 6]
            + 1.0925484305920792 * (x * z)[:, None] * coefficients[:, 7]
            + 0.5462742152960396 * (xx - yy)[:, None] * coefficients[:, 8]
        )
    if degree >= 3:
        xx, yy, zz = x.square(), y.square(), z.square()
        result = (
            result
            - 0.5900435899266435 * (y * (3.0 * xx - yy))[:, None] * coefficients[:, 9]
            + 2.890611442640554 * (x * y * z)[:, None] * coefficients[:, 10]
            - 0.4570457994644658 * (y * (4.0 * zz - xx - yy))[:, None]
            * coefficients[:, 11]
            + 0.3731763325901154 * (z * (2.0 * zz - 3.0 * xx - 3.0 * yy))[:, None]
            * coefficients[:, 12]
            - 0.4570457994644658 * (x * (4.0 * zz - xx - yy))[:, None]
            * coefficients[:, 13]
            + 1.445305721320277 * (z * (xx - yy))[:, None] * coefficients[:, 14]
            - 0.5900435899266435 * (x * (xx - 3.0 * yy))[:, None] * coefficients[:, 15]
        )
    return result
