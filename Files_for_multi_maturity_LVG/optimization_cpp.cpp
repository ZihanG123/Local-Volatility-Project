#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace py = pybind11;
using Vector = std::vector<double>;
using Coefficients = std::vector<std::array<double, 2>>;

struct Curve {
    Vector sigs;
    Vector nus;
    Coefficients c1;
    Coefficients c2;
};

using Point = std::array<double, 2>;
using LowerBound = std::vector<Point>;

Vector read_vector(py::handle object, const std::string& name) {
    using Array = py::array_t<double, py::array::c_style | py::array::forcecast>;
    Array array = Array::ensure(object);
    if (!array || array.ndim() != 1) {
        throw py::value_error(name + " must be a one-dimensional numeric array");
    }
    const auto view = array.unchecked<1>();
    Vector result(static_cast<size_t>(array.shape(0)));
    for (py::ssize_t i = 0; i < array.shape(0); ++i) {
        result[static_cast<size_t>(i)] = view(i);
    }
    return result;
}

std::vector<std::array<double, 3>> read_market(py::handle object) {
    using Array = py::array_t<double, py::array::c_style | py::array::forcecast>;
    Array array = Array::ensure(object);
    if (!array || array.ndim() != 2 || array.shape(1) != 3) {
        throw py::value_error("each bid/ask array must have shape (n, 3)");
    }
    const auto view = array.unchecked<2>();
    std::vector<std::array<double, 3>> result(
        static_cast<size_t>(array.shape(0)));
    for (py::ssize_t i = 0; i < array.shape(0); ++i) {
        result[static_cast<size_t>(i)] = {view(i, 0), view(i, 1), view(i, 2)};
    }
    return result;
}

Coefficients make_coefficients1(double S0, const Vector& sigs, const Vector& nus,
                                size_t split) {
    const size_t count = split;
    if (count == 0 || split + 1 >= nus.size()) {
        throw py::value_error("curve knots must straddle S0 with intervals on both sides");
    }
    Coefficients result;
    result.reserve(count);
    const double dist0 = nus[1] - nus[0];
    result.push_back({-std::exp(-dist0 / sigs[0]), std::exp(dist0 / sigs[0])});

    for (size_t i = 0; i + 1 < count; ++i) {
        const double a = result.back()[0] + result.back()[1];
        const double b = (-result.back()[0] + result.back()[1]) / sigs[i];
        const double dist = nus[i + 2] - nus[i + 1];
        result.push_back({
            0.5 * (a - sigs[i + 1] * b) * std::exp(-dist / sigs[i + 1]),
            0.5 * (a + sigs[i + 1] * b) * std::exp(dist / sigs[i + 1]),
        });
    }
    return result;
}

Coefficients make_coefficients2(double S0, const Vector& sigs, const Vector& nus,
                                size_t split) {
    (void)S0;
    const size_t count = sigs.size() - split;
    if (count == 0 || split == 0 || split + count >= nus.size()) {
        throw py::value_error("curve knots must straddle S0 with intervals on both sides");
    }
    Coefficients result(count);
    const double dist_end = nus.back() - nus[nus.size() - 2];
    result[count - 1] = {
        std::exp(dist_end / sigs.back()),
        -std::exp(-dist_end / sigs.back()),
    };

    for (size_t i = count - 1; i > 0; --i) {
        const double a = result[i][0] + result[i][1];
        const double b = (-result[i][0] + result[i][1]) / sigs[split + i];
        const double dist = nus[split + i] - nus[split + i - 1];
        result[i - 1] = {
            0.5 * (a - sigs[split + i - 1] * b) *
                std::exp(dist / sigs[split + i - 1]),
            0.5 * (a + sigs[split + i - 1] * b) *
                std::exp(-dist / sigs[split + i - 1]),
        };
    }
    return result;
}

Curve make_curve(double S0, Vector sigs, Vector nus) {
    if (nus.size() != sigs.size() + 1 || sigs.size() < 2 ||
        !std::is_sorted(nus.begin(), nus.end())) {
        throw py::value_error("each curve requires increasing nus and len(nus) == len(sigs) + 1");
    }
    const size_t split = static_cast<size_t>(
        std::lower_bound(nus.begin(), nus.end(), S0) - nus.begin());
    Curve curve{std::move(sigs), std::move(nus), {}, {}};
    curve.c1 = make_coefficients1(S0, curve.sigs, curve.nus, split);
    curve.c2 = make_coefficients2(S0, curve.sigs, curve.nus, split);

    const auto& left = curve.c1.back();
    const double v1 = left[0] + left[1];
    const double dv1 = (-left[0] + left[1]) / curve.sigs[split - 1];
    const auto& right = curve.c2.front();
    const double v2 = right[0] + right[1];
    const double dv2 = (-right[0] + right[1]) / curve.sigs[split];
    const double denominator = dv1 * v2 - v1 * dv2;
    const double lambda1 = v2 / denominator;
    const double lambda2 = v1 / denominator;
    for (auto& coeff : curve.c1) {
        coeff[0] *= lambda1;
        coeff[1] *= lambda1;
    }
    for (auto& coeff : curve.c2) {
        coeff[0] *= lambda2;
        coeff[1] *= lambda2;
    }
    return curve;
}

Vector evaluate(const Curve& curve, double S0, const Vector& strikes,
                bool call_value, bool derivative = false) {
    const size_t split = static_cast<size_t>(
        std::lower_bound(curve.nus.begin(), curve.nus.end(), S0) - curve.nus.begin());
    Vector values(strikes.size(), 0.0);
    for (size_t i = 0; i < strikes.size(); ++i) {
        const double strike = strikes[i];
        if (strike < curve.nus.front() || strike > curve.nus.back()) {
            continue;
        }
        if (strike < S0) {
            const auto end = curve.nus.begin() + static_cast<ptrdiff_t>(split + 1);
            size_t index = static_cast<size_t>(
                std::lower_bound(curve.nus.begin(), end, strike) - curve.nus.begin());
            index = std::clamp(index, size_t{1}, curve.c1.size());
            const size_t coeff_index = index - 1;
            const double dist = curve.nus[index] - strike;
            const double sigma = curve.sigs[coeff_index];
            const auto& coeff = curve.c1[coeff_index];
            if (derivative) {
                values[i] = (-coeff[0] * std::exp(dist / sigma) +
                             coeff[1] * std::exp(-dist / sigma)) / sigma;
                if (call_value) values[i] -= 1.0;
            } else {
                values[i] = coeff[0] * std::exp(dist / sigma) +
                            coeff[1] * std::exp(-dist / sigma);
                if (call_value) values[i] += S0 - strike;
            }
        } else {
            const auto begin = curve.nus.begin() + static_cast<ptrdiff_t>(split);
            size_t index = static_cast<size_t>(
                std::upper_bound(begin, curve.nus.end(), strike) - begin);
            index = std::clamp(index, size_t{1}, curve.c2.size());
            const size_t coeff_index = index - 1;
            const double dist = strike - curve.nus[split + coeff_index];
            const double sigma = curve.sigs[split + coeff_index];
            const auto& coeff = curve.c2[coeff_index];
            if (derivative) {
                values[i] = (-coeff[0] * std::exp(-dist / sigma) +
                             coeff[1] * std::exp(dist / sigma)) / sigma;
            } else {
                values[i] = coeff[0] * std::exp(-dist / sigma) +
                            coeff[1] * std::exp(dist / sigma);
            }
        }
    }
    return values;
}

LowerBound generate_lower_bound(const Curve& curve, double S0, int num_strikes) {
    Vector strikes(static_cast<size_t>(num_strikes));
    for (int i = 0; i < num_strikes; ++i) {
        const double u = static_cast<double>(i) / (num_strikes - 1);
        strikes[static_cast<size_t>(i)] = curve.nus.front() +
            (curve.nus.back() - curve.nus.front()) * u;
    }
    const Vector values = evaluate(curve, S0, strikes, true);
    const Vector slopes = evaluate(curve, S0, strikes, true, true);
    LowerBound result;
    result.reserve(static_cast<size_t>(num_strikes - 1));
    for (int i = 0; i + 1 < num_strikes; ++i) {
        const size_t j = static_cast<size_t>(i);
        const double slope_difference = slopes[j] - slopes[j + 1];
        const double intersection_strike =
            (values[j + 1] - values[j] + slopes[j] * strikes[j] -
             slopes[j + 1] * strikes[j + 1]) / slope_difference;
        const double intersection_value = values[j] + slopes[j] *
            (intersection_strike - strikes[j]);
        result.push_back({intersection_strike, intersection_value});
    }
    return result;
}

double objective(const Vector& sigs) {
    double result = 0.0;
    for (size_t i = 0; i + 1 < sigs.size(); ++i) {
        const double difference = sigs[i + 1] - sigs[i];
        result += difference * difference;
    }
    return result;
}

bool market_valid(const Curve& curve, double S0,
                  const std::vector<std::array<double, 3>>& market) {
    Vector strikes;
    strikes.reserve(market.size());
    for (const auto& row : market) strikes.push_back(row[0]);
    const Vector values = evaluate(curve, S0, strikes, true);
    for (size_t i = 0; i < market.size(); ++i) {
        if (values[i] < market[i][1] || values[i] > market[i][2]) return false;
    }
    return true;
}

bool monotonicity_valid(const std::vector<Curve>& curves,
                        const std::vector<LowerBound>& lower_bounds,
                        const Curve& candidate, double S0, size_t index,
                        int num_strikes, LowerBound& candidate_lower_bound) {
    if (curves.size() <= 1) return true;
    candidate_lower_bound = generate_lower_bound(candidate, S0, num_strikes);
    if (index > 0) {
        Vector strikes;
        Vector bound_values;
        for (const auto& point : candidate_lower_bound) {
            strikes.push_back(point[0]);
            bound_values.push_back(point[1]);
        }
        const Vector previous_values = evaluate(curves[index - 1], S0, strikes, true);
        for (size_t i = 0; i < strikes.size(); ++i) {
            if (!(previous_values[i] < bound_values[i])) return false;
        }
    }
    if (index + 1 < curves.size()) {
        Vector strikes;
        Vector bound_values;
        for (const auto& point : lower_bounds[index + 1]) {
            strikes.push_back(point[0]);
            bound_values.push_back(point[1]);
        }
        const Vector candidate_values = evaluate(candidate, S0, strikes, true);
        for (size_t i = 0; i < strikes.size(); ++i) {
            if (!(candidate_values[i] < bound_values[i])) return false;
        }
    }
    return true;
}

py::array_t<double> to_numpy(const Vector& values) {
    py::array_t<double> result(values.size());
    auto view = result.mutable_unchecked<1>();
    for (size_t i = 0; i < values.size(); ++i) view(static_cast<py::ssize_t>(i)) = values[i];
    return result;
}

py::array_t<double> to_numpy(const LowerBound& values) {
    py::array_t<double> result({static_cast<py::ssize_t>(values.size()), py::ssize_t{2}});
    auto view = result.mutable_unchecked<2>();
    for (size_t i = 0; i < values.size(); ++i) {
        view(static_cast<py::ssize_t>(i), 0) = values[i][0];
        view(static_cast<py::ssize_t>(i), 1) = values[i][1];
    }
    return result;
}

py::list coordinate_descent(double S0, py::sequence sigs_nus_compiled,
                            py::sequence bid_ask_compiled,
                            int num_iterations, int num_indices,
                            int num_strikes) {
    const size_t maturity_count = static_cast<size_t>(sigs_nus_compiled.size());
    if (bid_ask_compiled.size() != sigs_nus_compiled.size()) {
        throw py::value_error("sigs_nus_compiled and bid_ask_compiled must have equal lengths");
    }
    std::vector<Curve> curves;
    std::vector<std::vector<std::array<double, 3>>> markets;
    curves.reserve(maturity_count);
    markets.reserve(maturity_count);
    for (size_t i = 0; i < maturity_count; ++i) {
        py::sequence pair = py::reinterpret_borrow<py::sequence>(sigs_nus_compiled[i]);
        if (pair.size() != 2) throw py::value_error("each maturity must contain (sigs, nus)");
        curves.push_back(make_curve(S0, read_vector(pair[0], "sigs"),
                                    read_vector(pair[1], "nus")));
        markets.push_back(read_market(bid_ask_compiled[i]));
    }
    std::vector<LowerBound> lower_bounds(maturity_count);
    if (maturity_count > 1) {
        for (size_t i = 0; i < maturity_count; ++i) {
            lower_bounds[i] = generate_lower_bound(curves[i], S0, num_strikes);
        }
    }

    for (int iteration = 0; iteration < num_iterations; ++iteration) {
        for (size_t maturity = 0; maturity < maturity_count; ++maturity) {
            Curve& current = curves[maturity];
            double current_loss = objective(current.sigs);
            if (!market_valid(current, S0, markets[maturity])) {
                throw py::value_error("Initial sigs and nus do not satisfy market constraints");
            }
            if (num_indices < 0) throw py::value_error("num_indices must be nonnegative");
            const int length = static_cast<int>(current.sigs.size()) - num_indices + 1;
            if (length <= 0) continue;

            for (int coordinate = 0; coordinate < length; ++coordinate) {
                Vector gradient(current.sigs.size(), 0.0);
                if (current.sigs.size() > 2) {
                    for (size_t j = 1; j + 1 < current.sigs.size(); ++j) {
                        gradient[j] = (current.sigs[j - 1] + current.sigs[j + 1]) /
                            2.0 - current.sigs[j];
                    }
                }
                gradient.front() = current.sigs[1] - current.sigs[0];
                gradient.back() = current.sigs[current.sigs.size() - 2] -
                    current.sigs.back();
                const int start = coordinate % length;
                for (size_t j = 0; j < gradient.size(); ++j) {
                    if (static_cast<int>(j) < start ||
                        static_cast<int>(j) >= start + num_indices) {
                        gradient[j] = 0.0;
                    }
                }

                double alpha = 1.0;
                for (int backtrack = 0; backtrack < 5; ++backtrack) {
                    Vector test_sigs = current.sigs;
                    for (size_t j = 0; j < test_sigs.size(); ++j) {
                        test_sigs[j] += alpha * gradient[j];
                    }
                    const double test_loss = objective(test_sigs);
                    if (test_loss < current_loss) {
                        Curve candidate = make_curve(S0, test_sigs, current.nus);
                        if (market_valid(candidate, S0, markets[maturity])) {
                            LowerBound candidate_lower_bound;
                            if (monotonicity_valid(curves, lower_bounds, candidate,
                                                   S0, maturity, num_strikes,
                                                   candidate_lower_bound)) {
                                current = std::move(candidate);
                                current_loss = test_loss;
                                if (maturity_count > 1) {
                                    lower_bounds[maturity] =
                                        std::move(candidate_lower_bound);
                                }
                                break;
                            }
                        }
                    }
                    alpha *= 0.5;
                }
            }
        }
    }

    py::list result;
    for (const auto& curve : curves) {
        result.append(py::make_tuple(to_numpy(curve.sigs), to_numpy(curve.nus)));
    }
    return result;
}

PYBIND11_MODULE(_optimization_cpp, module) {
    module.doc() = "Native coordinate-descent optimizer for CompFi";
    module.def("coordinate_descent", &coordinate_descent,
               py::arg("S0"), py::arg("sigs_nus_compiled"),
               py::arg("bid_ask_compiled"), py::arg("num_iterations") = 1,
               py::arg("num_indices") = 1, py::arg("num_strikes") = 75);
}