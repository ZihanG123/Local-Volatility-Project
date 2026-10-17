import numpy as np

def load_theta(S, C, K, L, Kbar, prev_sigs=None, prev_nus=None):
    prev_c_vals1, prev_c_vals2 = None, None
    if prev_sigs is not None:
        prev_c_vals1, prev_c_vals2 = load_c_values(S, prev_sigs, prev_nus)

    def A_t(sigma, A, B, K_value):
        return 0.5 * (
            (A - sigma * B) * np.exp(-K_value / sigma)
            + (A + sigma * B) * np.exp(K_value / sigma)
        )

    def B_t(sigma, A, B, K_value):
        return (1 / (2 * sigma)) * (
            -(A - sigma * B) * np.exp(-K_value / sigma)
            + (A + sigma * B) * np.exp(K_value / sigma)
        )

    def bisect(function, low, high, tolerance=1e-12):
        if function(low) < 0:
            return low
        if function(high) > 0:
            return high
        while high - low > tolerance:
            mid = (low + high) / 2
            if mid == low or mid == high:
                return mid
            if function(mid) < 0:
                high = mid
            else:
                low = mid
        return (low + high) / 2

    def SigmaHat(A, B, K1, K2, V):
        def SigmaHat_obj(sigma):
            return A_t(sigma, A, B, K1) + B_t(sigma, A, B, K1) * K2 - V

        low = 1e-8
        high = 1.0
        while SigmaHat_obj(high) > 0:
            high *= 2
        return bisect(SigmaHat_obj, low, high)

    def SigmaTilde(sigma, A, B, K1, K2, V):
        At, Bt = A_t(sigma, A, B, K1), B_t(sigma, A, B, K1)

        def SigmaTilde_obj(sigma_t):
            return A_t(sigma_t, At, Bt, K2) - V

        low = 1e-8
        high = 1.0
        while SigmaTilde_obj(high) > 0:
            high *= 2
        return bisect(SigmaTilde_obj, low, high)

    def CalibrateTwoSigmas(A, B, K1, K2, V, B1):
        def TwoSigma_obj(sigma):
            sigma_t = SigmaTilde(sigma, A, B, K1, K2, V)
            At, Bt = A_t(sigma, A, B, K1), B_t(sigma, A, B, K1)
            return B1 - B_t(sigma_t, At, Bt, K2)

        low = SigmaHat(A, B, K1, K2, V)
        high = low + 1
        while TwoSigma_obj(high) > 0:
            high *= 2
        result = bisect(TwoSigma_obj, low, high)
        return result, SigmaTilde(result, A, B, K1, K2, V)

    def CalibrateThreeSigmas(A, B, K1, K2, K3, W, V, B1):
        sigma1 = SigmaHat(A, B, K1, K2, W)
        At, Bt = A_t(sigma1, A, B, K1), B_t(sigma1, A, B, K1)
        sigma2, sigma3 = CalibrateTwoSigmas(At, Bt, K2, K3, V, B1)
        return sigma1, sigma2, sigma3

    def previous_V(strike):
        return V_K(
            S,
            prev_sigs,
            prev_nus,
            prev_c_vals1,
            prev_c_vals2,
            np.array([strike]),
        )[0]

    def Intersection_obj(K_value, A, B, x, side="left"):
        if side == "left":
            return (
                A
                + B * (x - K_value)
                - previous_V(L + x)
            )
        else:
            return (
                A
                + B * (x - K_value)
                - previous_V(Kbar - x)
            )

    i0 = np.searchsorted(K, S)
    Lt1 = (
        C[i0 - 1]
        + (S - K[i0 - 1])
        * (C[i0 - 1] - C[i0 - 2])
        / (K[i0 - 1] - K[i0 - 2])
    )
    Lt2 = (
        C[i0]
        + (S - K[i0]) * (C[i0 + 1] - C[i0])
        / (K[i0 + 1] - K[i0])
    )
    Lt = max(Lt1, Lt2)
    Ut = (
        C[i0 - 1] * (K[i0] - S) / (K[i0] - K[i0 - 1])
        + C[i0] * (S - K[i0 - 1]) / (K[i0] - K[i0 - 1])
    )
    Vtemp = 0.5 * Lt + 0.5 * Ut

    K1 = np.concatenate(([L], K[:i0], [S]))
    V1 = np.concatenate(([0], C[:i0] - S + K[:i0], [Vtemp]))
    K2 = np.concatenate(([0], Kbar - K[i0:][::-1], [Kbar - S]))
    V2 = np.concatenate(([0], C[i0:][::-1], [Vtemp]))

    def CalibrationAlgorithm(K0, K1, A, V, B, B1, side="left"):
        nu = []
        sigmas = []
        flag = False
        if prev_sigs is not None:
            cross1 = bisect(
                lambda x: Intersection_obj(K0, A, B, x, side=side),
                K0 + 1e-8,
                K1 - 1e-8,
            )
            cross2 = bisect(
                lambda x: -Intersection_obj(K1, V, B1, x, side=side),
                K0 + 1e-8,
                K1 - 1e-8,
            )
            flag = cross1 < cross2
        if flag:
            w1 = min((K1 - K0) / 3, cross1 - K0)
            w2 = max((K1 - K0) * 2 / 3, cross2 - K0)
            W = V + B1 * (w2 + K0 - K1)
            sigma1, sigma2, sigma3 = CalibrateThreeSigmas(
                A, B, w1, w2 - w1, K1 - K0 - w2, W, V, B1
            )
            nu.extend((K0, K0 + w1, K0 + w2))
            sigmas.extend((sigma1, sigma2, sigma3))
        else:
            w = (V + B * K0 - A - B1 * K1) / (B - B1)
            sigma1, sigma2 = CalibrateTwoSigmas(
                A, B, w - K0, K1 - w, V, B1
            )
            nu.extend((K0, w))
            sigmas.extend((sigma1, sigma2))
        return nu, sigmas

    def Interpolate(K_values, V_values, side="left"):
        nu = []
        sigmas = []
        B = 0.5 * (V_values[1] - V_values[0]) / (K_values[1] - K_values[0])

        A = V_values[0]
        V1 = V_values[1]
        V2 = V_values[2]

        for i in range(len(K_values) - 2):
            K0 = K_values[i]
            K1 = K_values[i + 1]
            K2 = K_values[i + 2]
            h_left = K1 - K0
            h_right = K2 - K1
            s_left = (V1 - A) / h_left
            s_right = (V2 - V1) / h_right
            B1 = (h_right * s_left + h_left * s_right) / (h_left + h_right)
            nus_new, sigmas_new = CalibrationAlgorithm(K0, K1, A, V1, B, B1, side=side)
            nu.extend(nus_new)
            sigmas.extend(sigmas_new)
            B = B1
            A = V1
            V1 = V2
            if i + 3 < len(V_values): V2 = V_values[i + 3]
        return nu, sigmas, B

    nu1, sigmas1, B1_last = Interpolate(K1, V1, side="left")
    nu2, sigmas2, B2_last = Interpolate(K2, V2, side="right")
    delta1 = 0.5
    rho = (
        delta1
        + delta1 * (V2[-2] - V2[-1]) / (K2[-1] - K2[-2])
        + (1.0 - delta1) * (V1[-1] - V1[-2]) / (K1[-1] - K1[-2])
    )

    nus_new, sigmas_new = CalibrationAlgorithm(
        K1[-2], K1[-1], V1[-2], V1[-1], B1_last, rho, side="left"
    )
    nu1.extend(nus_new)
    sigmas1.extend(sigmas_new)

    nus_new, sigmas_new = CalibrationAlgorithm(
        K2[-2], K2[-1], V2[-2], V2[-1], B2_last, 1 - rho, side="right"
    )
    nu2.extend(nus_new)
    sigmas2.extend(sigmas_new)

    sigs = np.concatenate([sigmas1, sigmas2[::-1]])
    nus = np.concatenate([nu1, [S], Kbar - np.array(nu2[::-1])])
    return sigs, nus


def load_c_values(S0, sigs, nus):
    R1 = np.searchsorted(nus, S0)
    nus1, sigs1 = nus[: R1 + 1], sigs[:R1]
    nus2, sigs2 = nus[R1:], sigs[R1:]

    c_vals1 = []
    dist0 = nus1[1] - nus1[0]
    c_vals1.append(
        np.stack([-np.exp(-dist0 / sigs1[0]), np.exp(dist0 / sigs1[0])])
    )

    for i in range(len(sigs1) - 1):
        A = c_vals1[-1][0] + c_vals1[-1][1]
        B = (1 / sigs1[i]) * (-c_vals1[-1][0] + c_vals1[-1][1])
        dist = nus1[i + 2] - nus1[i + 1]
        c_vals1.append(
            np.stack(
                [
                    0.5 * (A - sigs1[i + 1] * B) * np.exp(-dist / sigs1[i + 1]),
                    0.5 * (A + sigs1[i + 1] * B) * np.exp(dist / sigs1[i + 1]),
                ]
            )
        )

    c_vals2 = [None] * len(sigs2)
    dist_end = nus2[-1] - nus2[-2]
    c_vals2[-1] = np.stack(
        [np.exp(dist_end / sigs2[-1]), -np.exp(-dist_end / sigs2[-1])]
    )

    for i in range(len(sigs2) - 1, 0, -1):
        A = c_vals2[i][0] + c_vals2[i][1]
        B = (1 / sigs2[i]) * (-c_vals2[i][0] + c_vals2[i][1])
        dist = nus2[i] - nus2[i - 1]
        c_vals2[i - 1] = np.stack(
            [
                0.5 * (A - sigs2[i - 1] * B) * np.exp(dist / sigs2[i - 1]),
                0.5 * (A + sigs2[i - 1] * B) * np.exp(-dist / sigs2[i - 1]),
            ]
        )

    c_vals1_tensor = np.stack(c_vals1)
    c_vals2_tensor = np.stack(c_vals2)
    v1 = c_vals1_tensor[-1, 0] + c_vals1_tensor[-1, 1]
    Dk_v1 = (1 / sigs1[-1]) * (-c_vals1_tensor[-1, 0] + c_vals1_tensor[-1, 1])
    v2 = c_vals2_tensor[0, 0] + c_vals2_tensor[0, 1]
    Dk_v2 = (1 / sigs2[0]) * (-c_vals2_tensor[0, 0] + c_vals2_tensor[0, 1])
    denom = Dk_v1 * v2 - v1 * Dk_v2
    lam1, lam2 = v2 / denom, v1 / denom
    return c_vals1_tensor * lam1, c_vals2_tensor * lam2


def V_K(S0, sigs, nus, c_v1, c_v2, strikes, type="V"):
    R1 = np.searchsorted(nus, S0)
    nus1, sigs1 = nus[: R1 + 1], sigs[:R1]
    nus2, sigs2 = nus[R1:], sigs[R1:]

    values = np.zeros(strikes.shape, dtype=float)
    in_domain = (strikes >= nus[0]) & (strikes <= nus[-1])
    left_mask = in_domain & (strikes < S0)
    right_mask = in_domain & ~left_mask

    if np.any(left_mask):
        left_strikes = strikes[left_mask]
        idx = np.clip(np.searchsorted(nus1, left_strikes), 1, len(sigs1))
        dist = nus1[idx] - left_strikes
        sigma = sigs1[idx - 1]
        values[left_mask] = (
            c_v1[idx - 1, 0] * np.exp(dist / sigma)
            + c_v1[idx - 1, 1] * np.exp(-dist / sigma)
        )
        if type == "C": values[left_mask] += (S0 - left_strikes)

    if np.any(right_mask):
        right_strikes = strikes[right_mask]
        idx = np.clip( np.searchsorted(nus2, right_strikes, side="right"), 1, len(sigs2))
        dist = right_strikes - nus2[idx - 1]
        sigma = sigs2[idx - 1]
        values[right_mask] = (
            c_v2[idx - 1, 0] * np.exp(-dist / sigma)
            + c_v2[idx - 1, 1] * np.exp(dist / sigma)
        )

    return values

def V_K_prime(S0, sigs, nus, c_v1, c_v2, strikes, type="V"):
    R1 = np.searchsorted(nus, S0)
    nus1, sigs1 = nus[: R1 + 1], sigs[:R1]
    nus2, sigs2 = nus[R1:], sigs[R1:]

    values = np.zeros(strikes.shape, dtype=float)
    in_domain = (strikes >= nus[0]) & (strikes <= nus[-1])
    left_mask = in_domain & (strikes < S0)
    right_mask = in_domain & ~left_mask

    if np.any(left_mask):
        left_strikes = strikes[left_mask]
        idx = np.clip(np.searchsorted(nus1, left_strikes), 1, len(sigs1))
        dist = nus1[idx] - left_strikes
        sigma = sigs1[idx - 1]
        values[left_mask] = (1/sigma) * (
            c_v1[idx - 1, 0] * -np.exp(dist / sigma)
            + c_v1[idx - 1, 1] * np.exp(-dist / sigma)
        )
        if type == "C": values[left_mask] -= 1

    if np.any(right_mask):
        right_strikes = strikes[right_mask]
        idx = np.clip(np.searchsorted(nus2, right_strikes, side="right"), 1, len(sigs2))
        dist = right_strikes - nus2[idx - 1]
        sigma = sigs2[idx - 1]
        values[right_mask] = (1/sigma) * (
            c_v2[idx - 1, 0] * -np.exp(-dist / sigma)
            + c_v2[idx - 1, 1] * np.exp(dist / sigma)
        )
    return values

