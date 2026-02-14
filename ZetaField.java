/*
 * ═══════════════════════════════════════════════════════════════
 * ζ-Field Model — Java Implementation
 * ═══════════════════════════════════════════════════════════════
 * Ξ_t(x,y) = Norm_[0,100] [ V + F + C + B + S + A + H + J + Q + T ]
 *
 * A composite mathematical field that models asset price behavior by
 * combining 10 component functions over a 2D phase space (Φ(t), Φ(t-1)).
 *
 * Outputs CSV files for external visualization (gnuplot, Python, Plotly, etc.)
 *
 * NOTE: The actual ζ-Field equations are proprietary. This is an inspired
 * approximation using known quantitative-finance concepts.
 * ═══════════════════════════════════════════════════════════════
 */

import java.io.*;
import java.util.Random;

public class ZetaField {

    // ─────────────────────────────────────────────
    // 1. COMPONENT FUNCTIONS
    // ─────────────────────────────────────────────

    /** Radial potential — localized Gaussian well centered at (aLoc, bLoc). */
    static double V(double r, double aLoc, double bLoc, double depth, double sigma) {
        return depth * Math.exp(-r * r / (2.0 * sigma * sigma));
    }

    static double V(double r) {
        return V(r, 0.0, 0.0, 30.0, 1.5);
    }

    /** Marginal force — asymmetric sigmoid representing directional pressure. */
    static double F(double x, double aLoc, double k, double shift) {
        return k / (1.0 + Math.exp(-(x - aLoc) * 3.0)) - shift;
    }

    static double F(double x) {
        return F(x, 0.0, 2.0, 0.5);
    }

    /** Coupling field — interaction between Φ(t) and Φ(t-1). */
    static double C(double x, double y, double b) {
        return b * x * y / (1.0 + Math.abs(x * y));
    }

    static double C(double x, double y) {
        return C(x, y, 0.6);
    }

    /** Boundary feedback — self-referential term dampening extremes. */
    static double B(double r, double xiPrev, double t, double gamma) {
        return -gamma * r * r * Math.sin(t * 0.1) * (1.0 + 0.01 * xiPrev);
    }

    static double B(double r, double xiPrev, double t) {
        return B(r, xiPrev, t, 0.05);
    }

    /** Skewness surface — captures asymmetry in returns. */
    static double S(double x, double y, double chi) {
        return chi * (x * x * x - y * y * y) / (1.0 + x * x + y * y);
    }

    static double S(double x, double y) {
        return S(x, y, 0.3);
    }

    /** Amplitude modulator — scales overall field intensity. */
    static double A(double x, double y, double alpha) {
        return alpha * Math.exp(-(x * x + y * y) / 20.0);
    }

    static double A(double x, double y) {
        return A(x, y, 1.0);
    }

    /** Harmonic oscillator — captures periodic / cyclical patterns. */
    static double H(double r, double x, double y, double xiPrev, double t,
                    double omega, double amp) {
        double phiAngle = Math.atan2(y, x);
        return amp * Math.cos(omega * t + phiAngle) * Math.exp(-0.1 * r);
    }

    static double H(double r, double x, double y, double xiPrev, double t) {
        return H(r, x, y, xiPrev, t, 0.5, 5.0);
    }

    /** Jump diffusion — models fat tails and sudden jumps. */
    static double J(double r, double xiPrev, double alpha, double lam) {
        return lam * Math.exp(-alpha * r) * (1.0 + 0.01 * xiPrev);
    }

    static double J(double r, double xiPrev) {
        return J(r, xiPrev, 1.0, 0.3);
    }

    /** Quantile regressor — maps field to quantile positions. */
    static double Q(double r, double xiPrev, double q) {
        return q * Math.log1p(r) * Math.tanh(0.01 * xiPrev);
    }

    static double Q(double r, double xiPrev) {
        return Q(r, xiPrev, 0.5);
    }

    /** Temporal drift — time-varying component. */
    static double T(double x, double y, double t, double drift) {
        return drift * t * Math.exp(-(x * x + y * y) / 30.0);
    }

    static double T(double x, double y, double t) {
        return T(x, y, t, 0.02);
    }


    // ─────────────────────────────────────────────
    // 2. COMPOSITE FIELD Ξ_t(x, y)
    // ─────────────────────────────────────────────

    /**
     * Compute the raw (un-normalized) composite field value at a single point.
     */
    static double xiFieldRaw(double x, double y, double t, double xiPrev) {
        double r = Math.sqrt(x * x + y * y);
        return V(r) + F(x) + C(x, y) + B(r, xiPrev, t) + S(x, y)
                + A(x, y) + H(r, x, y, xiPrev, t) + J(r, xiPrev)
                + Q(r, xiPrev) + T(x, y, t);
    }

    /**
     * Compute the full ζ-field surface on a grid and normalize to [0, 100].
     *
     * @param N       grid resolution (N × N)
     * @param lo      lower bound of coordinate range
     * @param hi      upper bound of coordinate range
     * @param t       time step
     * @param xiPrev  previous field value
     * @return 2D array [N][N] of normalized values
     */
    static double[][] xiFieldGrid(int N, double lo, double hi,
                                   double t, double xiPrev) {
        double[][] Z = new double[N][N];
        double step = (hi - lo) / (N - 1);
        double rmin = Double.MAX_VALUE, rmax = -Double.MAX_VALUE;

        // First pass: compute raw values
        for (int i = 0; i < N; i++) {
            double x = lo + i * step;
            for (int j = 0; j < N; j++) {
                double y = lo + j * step;
                double val = xiFieldRaw(x, y, t, xiPrev);
                Z[i][j] = val;
                if (val < rmin) rmin = val;
                if (val > rmax) rmax = val;
            }
        }

        // Second pass: normalize to [0, 100]
        double range = rmax - rmin;
        if (range < 1e-12) {
            for (int i = 0; i < N; i++)
                for (int j = 0; j < N; j++)
                    Z[i][j] = 50.0;
        } else {
            for (int i = 0; i < N; i++)
                for (int j = 0; j < N; j++)
                    Z[i][j] = 100.0 * (Z[i][j] - rmin) / range;
        }
        return Z;
    }

    /**
     * Normalize a single value given known min/max of the field.
     */
    static double normalizeValue(double raw, double rmin, double rmax) {
        double range = rmax - rmin;
        if (range < 1e-12) return 50.0;
        return 100.0 * (raw - rmin) / range;
    }


    // ─────────────────────────────────────────────
    // 3. CSV SURFACE OUTPUT
    // ─────────────────────────────────────────────

    static void writeSurfaceCsv(String path) throws IOException {
        int N = 200;
        double lo = -5.0, hi = 5.0;
        double[][] Z = xiFieldGrid(N, lo, hi, 5.0, 50.0);
        double step = (hi - lo) / (N - 1);

        try (PrintWriter pw = new PrintWriter(new FileWriter(path))) {
            pw.println("x,y,z");
            for (int i = 0; i < N; i++) {
                double x = lo + i * step;
                for (int j = 0; j < N; j++) {
                    double y = lo + j * step;
                    pw.printf("%.6f,%.6f,%.6f%n", x, y, Z[i][j]);
                }
            }
        }
        System.out.printf("[✓] Surface CSV saved → %s%n", path);
    }


    // ─────────────────────────────────────────────
    // 4. PORTFOLIO BACKTEST (synthetic demo)
    // ─────────────────────────────────────────────

    /** Generate synthetic daily prices using geometric Brownian motion + jumps. */
    static double[] generateSyntheticPrices(int n, long seed) {
        Random rng = new Random(seed);
        double dt = 1.0 / 252.0;
        double mu = 0.06, sigma = 0.18;
        double[] prices = new double[n];
        prices[0] = 100.0;

        for (int i = 1; i < n; i++) {
            double jump = (rng.nextDouble() < 0.03)
                    ? rng.nextGaussian() * 0.03 : 0.0;
            prices[i] = prices[i - 1] * Math.exp(
                    (mu - 0.5 * sigma * sigma) * dt
                    + sigma * Math.sqrt(dt) * rng.nextGaussian()
                    + jump
            );
        }
        return prices;
    }

    /**
     * Compute simplified ζ-field trading signal.
     * Returns +1 (long), 0 (flat), -1 (short).
     */
    static double[] zetaSignal(double[] prices, int lookback,
                               double sigmaThreshold) {
        int n = prices.length;
        double[] signals = new double[n];

        for (int i = lookback; i < n; i++) {
            // Compute log returns for window
            double[] rets = new double[lookback - 1];
            for (int j = 0; j < lookback - 1; j++) {
                rets[j] = Math.log(prices[i - lookback + j + 1]
                        / prices[i - lookback + j]);
            }
            if (rets.length < 2) continue;

            double phiT  = rets[rets.length - 1] * 100.0;
            double phiT1 = rets[rets.length - 2] * 100.0;

            // Compute field value for this single point
            double raw = xiFieldRaw(phiT, phiT1, (double) i, 50.0);
            // For single-point evaluation, normalize using a rough global range
            // (deterministic approximation: the surface range for the default params)
            double fieldVal = Math.max(0.0, Math.min(100.0, raw));

            // Use a simpler threshold on the raw value since single-point
            // normalization isn't meaningful. Re-center around the raw midpoint.
            // Instead, compute a quick local normalization:
            double rawCenter = xiFieldRaw(0, 0, (double) i, 50.0);
            double delta = raw - rawCenter;

            if (delta > sigmaThreshold * 10.0) {
                signals[i] = 1.0;   // bullish regime
            } else if (delta < -sigmaThreshold * 10.0) {
                signals[i] = -1.0;  // bearish regime
            } else {
                signals[i] = 0.0;   // neutral
            }
        }
        return signals;
    }

    /** Compute portfolio equity curve from signals. */
    static double[] runBacktest(double[] prices, double[] signals,
                                double initialCapital) {
        int n = prices.length;
        double[] equity = new double[n];
        equity[0] = initialCapital;
        for (int i = 1; i < n; i++) {
            double dailyRet = prices[i] / prices[i - 1] - 1.0;
            equity[i] = equity[i - 1] * (1.0 + signals[i - 1] * dailyRet);
        }
        return equity;
    }

    /** Maximum drawdown (%). */
    static double maxDrawdown(double[] equity) {
        double peak = equity[0];
        double maxDd = 0.0;
        for (double v : equity) {
            if (v > peak) peak = v;
            double dd = (peak - v) / peak;
            if (dd > maxDd) maxDd = dd;
        }
        return maxDd * 100.0;
    }

    /** Write backtest results to CSV. */
    static void writeBacktestCsv(String path) throws IOException {
        int n = 3000;
        double[] prices = generateSyntheticPrices(n, 42);
        double[] years = new double[n];
        for (int i = 0; i < n; i++)
            years[i] = 2006.0 + (2018.0 - 2006.0) * i / (n - 1);

        double[] bhEquity = new double[n];
        for (int i = 0; i < n; i++)
            bhEquity[i] = 100.0 * prices[i] / prices[0];

        double[] sigZ = zetaSignal(prices, 60, 0.3);
        double[] eqZ  = runBacktest(prices, sigZ, 100.0);

        double[] sigE = zetaSignal(prices, 60, 0.4);
        double[] eqE  = runBacktest(prices, sigE, 100.0);

        try (PrintWriter pw = new PrintWriter(new FileWriter(path))) {
            pw.println("year,buy_and_hold,zeta_sigma03,eclipse_sigma04");
            for (int i = 0; i < n; i++) {
                pw.printf("%.4f,%.6f,%.6f,%.6f%n",
                        years[i], bhEquity[i], eqZ[i], eqE[i]);
            }
        }

        System.out.printf("[✓] Backtest CSV saved → %s%n", path);
        System.out.printf("    BUY_&_HOLD    final=$%.0f  max_dd=%.1f%%%n",
                bhEquity[n - 1], maxDrawdown(bhEquity));
        System.out.printf("    ZETA_σ.3      final=$%.0f  max_dd=%.1f%%%n",
                eqZ[n - 1], maxDrawdown(eqZ));
        System.out.printf("    ECLIPSE_σ.4   final=$%.0f  max_dd=%.1f%%%n",
                eqE[n - 1], maxDrawdown(eqE));
    }


    // ─────────────────────────────────────────────
    // 5. MAIN
    // ─────────────────────────────────────────────

    public static void main(String[] args) {
        System.out.println("=".repeat(60));
        System.out.println("  ζ-Field Model — Java Implementation");
        System.out.println("=".repeat(60));
        System.out.println();

        try {
            System.out.println("[1/2] Computing 3D ζ-Field surface → CSV …");
            writeSurfaceCsv("zeta_field_3d_java.csv");

            System.out.println("[2/2] Running portfolio backtest …");
            writeBacktestCsv("zeta_backtest_java.csv");

            System.out.println();
            System.out.println("Done. Output files:");
            System.out.println("  • zeta_field_3d_java.csv   — 3D field surface data");
            System.out.println("  • zeta_backtest_java.csv   — portfolio backtest data");
        } catch (IOException e) {
            System.err.println("Error: " + e.getMessage());
            e.printStackTrace();
        }
    }
}
