#!/usr/bin/env Rscript
# ═══════════════════════════════════════════════════════════════
# ζ-Field Model — R Implementation
# ═══════════════════════════════════════════════════════════════
# Ξ_t(x,y) = Norm_[0,100] [ V + F + C + B + S + A + H + J + Q + T ]
#
# A composite mathematical field that models asset price behavior by
# combining 10 component functions over a 2D phase space (Φ(t), Φ(t-1)).
#
# NOTE: The actual ζ-Field equations are proprietary. This is an inspired
# approximation using known quantitative-finance concepts.
# ═══════════════════════════════════════════════════════════════

suppressPackageStartupMessages({
  if (!requireNamespace("plotly", quietly = TRUE))
    install.packages("plotly", repos = "https://cloud.r-project.org")
  if (!requireNamespace("ggplot2", quietly = TRUE))
    install.packages("ggplot2", repos = "https://cloud.r-project.org")
  if (!requireNamespace("htmlwidgets", quietly = TRUE))
    install.packages("htmlwidgets", repos = "https://cloud.r-project.org")
  library(plotly)
  library(ggplot2)
  library(htmlwidgets)
})

# ─────────────────────────────────────────────
# 1. COMPONENT FUNCTIONS
# ─────────────────────────────────────────────

#' Radial potential — localized Gaussian well centered at (a_loc, b_loc).
V <- function(r, a_loc = 0.0, b_loc = 0.0, depth = 30.0, sigma = 1.5) {
  depth * exp(-r^2 / (2 * sigma^2))
}

#' Marginal force — asymmetric sigmoid representing directional pressure.
F_func <- function(x, a_loc = 0.0, k = 2.0, shift = 0.5) {
  k / (1 + exp(-(x - a_loc) * 3)) - shift
}

#' Coupling field — interaction between Φ(t) and Φ(t-1).
C_func <- function(x, y, b = 0.6) {
  b * x * y / (1 + abs(x * y))
}

#' Boundary feedback — self-referential term dampening extremes.
B_func <- function(r, xi_prev, t, gamma = 0.05) {
  -gamma * r^2 * sin(t * 0.1) * (1 + 0.01 * xi_prev)
}

#' Skewness surface — captures asymmetry in returns.
S_func <- function(x, y, chi = 0.3) {
  chi * (x^3 - y^3) / (1 + x^2 + y^2)
}

#' Amplitude modulator — scales overall field intensity.
A_func <- function(x, y, alpha = 1.0) {
  alpha * exp(-(x^2 + y^2) / 20)
}

#' Harmonic oscillator — captures periodic / cyclical patterns.
H_func <- function(r, x, y, xi_prev, t, omega = 0.5, amp = 5.0) {
  phi_angle <- atan2(y, x)
  amp * cos(omega * t + phi_angle) * exp(-0.1 * r)
}

#' Jump diffusion — models fat tails and sudden jumps.
J_func <- function(r, xi_prev, alpha = 1.0, lam = 0.3) {
  lam * exp(-alpha * r) * (1 + 0.01 * xi_prev)
}

#' Quantile regressor — maps field to quantile positions.
Q_func <- function(r, xi_prev, q = 0.5) {
  q * log1p(r) * tanh(0.01 * xi_prev)
}

#' Temporal drift — time-varying component.
T_func <- function(x, y, t, drift = 0.02) {
  drift * t * exp(-(x^2 + y^2) / 30)
}


# ─────────────────────────────────────────────
# 2. COMPOSITE FIELD Ξ_t(x, y)
# ─────────────────────────────────────────────

#' Compute Ξ_t(x,y) — the full composite field, normalized to [0, 100].
#'
#' @param x,y   Numeric vectors/matrices — phase-space coords (Φ(t), Φ(t-1))
#' @param t     Numeric — time step
#' @param xi_prev Numeric — previous field value (self-referential feedback)
#' @param params  Named list — optional parameter overrides
#' @return Numeric vector/matrix normalized to [0, 100]
xi_field <- function(x, y, t = 1.0, xi_prev = 50.0, params = list()) {
  a_loc <- ifelse(is.null(params$a_loc), 0.0, params$a_loc)
  b_loc <- ifelse(is.null(params$b_loc), 0.0, params$b_loc)
  b_val <- ifelse(is.null(params$b), 0.6, params$b)
  chi   <- ifelse(is.null(params$chi), 0.3, params$chi)
  alpha <- ifelse(is.null(params$alpha), 1.0, params$alpha)

  r <- sqrt(x^2 + y^2)

  raw <- (
    V(r, a_loc, b_loc) +
    F_func(x, a_loc) +
    C_func(x, y, b_val) +
    B_func(r, xi_prev, t) +
    S_func(x, y, chi) +
    A_func(x, y, alpha) +
    H_func(r, x, y, xi_prev, t) +
    J_func(r, xi_prev) +
    Q_func(r, xi_prev) +
    T_func(x, y, t)
  )

  # Normalize to [0, 100]
  rmin <- min(raw, na.rm = TRUE)
  rmax <- max(raw, na.rm = TRUE)
  if ((rmax - rmin) < 1e-12) {
    return(rep(50.0, length(raw)))
  }
  100.0 * (raw - rmin) / (rmax - rmin)
}


# ─────────────────────────────────────────────
# 3. 3D SURFACE VISUALIZATION (Interactive Plotly)
# ─────────────────────────────────────────────

plot_3d_surface <- function(save_path = "zeta_field_3d_r.html") {
  N <- 200
  lin <- seq(-5, 5, length.out = N)
  grid <- expand.grid(x = lin, y = lin)
  X <- matrix(grid$x, nrow = N, ncol = N)
  Y <- matrix(grid$y, nrow = N, ncol = N)
  Z <- matrix(xi_field(grid$x, grid$y, t = 5.0, xi_prev = 50.0),
              nrow = N, ncol = N)

  # Custom colorscale matching the Python version
  cscale <- list(
    list(0.00, "#050520"),
    list(0.20, "#0a1a5c"),
    list(0.40, "#1e6fa0"),
    list(0.55, "#5ec8e0"),
    list(0.70, "#b8eaf0"),
    list(0.85, "#f5f7a8"),
    list(0.95, "#f0a030"),
    list(1.00, "#e03010")
  )

  p <- plot_ly(
    x = lin, y = lin, z = Z,
    type = "surface",
    colorscale = cscale,
    showscale = TRUE,
    colorbar = list(title = list(text = "PERIODICITY", font = list(color = "white")),
                    tickfont = list(color = "white"))
  ) %>%
    layout(
      title = list(
        text = "\u03b6-Field  \u039e\u209c(x,y) = Norm\u2080\u208c\u2081\u2080\u2080[V+F+C+B+S+A+H+J+Q+T]",
        font = list(color = "white", size = 16)
      ),
      paper_bgcolor = "black",
      plot_bgcolor  = "black",
      scene = list(
        xaxis = list(title = "\u03a6(t)",   color = "white", gridcolor = "#333"),
        yaxis = list(title = "\u03a6(t-1)", color = "white", gridcolor = "#333"),
        zaxis = list(title = "\u03a6 DENSITY", color = "white", gridcolor = "#333"),
        bgcolor = "black",
        camera = list(eye = list(x = 1.5, y = -1.5, z = 1.0))
      )
    )

  htmlwidgets::saveWidget(p, file = save_path, selfcontained = TRUE)
  cat(sprintf("[✓] 3D surface saved → %s\n", save_path))
}


# ─────────────────────────────────────────────
# 4. PORTFOLIO BACKTEST (synthetic demo)
# ─────────────────────────────────────────────

#' Generate synthetic daily prices using geometric Brownian motion + jumps.
generate_synthetic_prices <- function(n = 3000, seed = 42) {
  set.seed(seed)
  dt    <- 1.0 / 252.0
  mu    <- 0.06
  sigma <- 0.18
  prices <- numeric(n)
  prices[1] <- 100.0

  for (i in 2:n) {
    jump <- ifelse(runif(1) < 0.03, rnorm(1, 0, 0.03), 0.0)
    prices[i] <- prices[i - 1] * exp(
      (mu - 0.5 * sigma^2) * dt +
      sigma * sqrt(dt) * rnorm(1) +
      jump
    )
  }
  prices
}

#' Compute a simplified ζ-field trading signal.
#' Returns +1 (long), 0 (flat), -1 (short).
zeta_signal <- function(prices, lookback = 60, sigma_threshold = 0.3) {
  n <- length(prices)
  signals <- numeric(n)
  for (i in (lookback + 1):n) {
    window <- prices[(i - lookback):(i - 1)]
    rets   <- diff(log(window))

    if (length(rets) < 2) next

    phi_t  <- rets[length(rets)]     * 100.0
    phi_t1 <- rets[length(rets) - 1] * 100.0

    field_val <- xi_field(phi_t, phi_t1, t = as.numeric(i), xi_prev = 50.0)

    if (length(field_val) > 1) field_val <- field_val[1]

    if (field_val > (50.0 + sigma_threshold * 30.0)) {
      signals[i] <- 1.0
    } else if (field_val < (50.0 - sigma_threshold * 30.0)) {
      signals[i] <- -1.0
    } else {
      signals[i] <- 0.0
    }
  }
  signals
}

#' Compute portfolio equity curve from signals.
run_backtest <- function(prices, signals, initial_capital = 100.0) {
  n <- length(prices)
  equity <- numeric(n)
  equity[1] <- initial_capital
  for (i in 2:n) {
    daily_ret <- prices[i] / prices[i - 1] - 1.0
    equity[i] <- equity[i - 1] * (1.0 + signals[i - 1] * daily_ret)
  }
  equity
}

#' Maximum drawdown (%).
max_dd <- function(eq) {
  peak <- cummax(eq)
  dd   <- (peak - eq) / peak
  max(dd, na.rm = TRUE) * 100.0
}

#' Produce the backtest comparison chart.
plot_backtest <- function(save_path = "zeta_backtest_r.png") {
  prices <- generate_synthetic_prices(n = 3000)
  n <- length(prices)
  years <- seq(2006, 2018, length.out = n)

  # Buy & Hold
  bh_equity <- 100.0 * prices / prices[1]

  # ZETA_σ.3
  sig_z <- zeta_signal(prices, sigma_threshold = 0.3)
  eq_z  <- run_backtest(prices, sig_z)

  # ECLIPSE_σ.4
  sig_e <- zeta_signal(prices, sigma_threshold = 0.4)
  eq_e  <- run_backtest(prices, sig_e)

  bh_final <- tail(bh_equity, 1)
  z_final  <- tail(eq_z, 1)
  e_final  <- tail(eq_e, 1)

  df <- data.frame(
    year = rep(years, 3),
    value = c(bh_equity, eq_z, eq_e),
    strategy = rep(
      c(sprintf("BUY_&_HOLD  $%.0f (%.0f%%)", bh_final, bh_final - 100),
        sprintf("ZETA_σ.3  $%.0f (%.0f%%)",    z_final,  z_final  - 100),
        sprintf("ECLIPSE_σ.4  $%.0f (%.0f%%)", e_final,  e_final  - 100)),
      each = n
    )
  )

  p <- ggplot(df, aes(x = year, y = value, colour = strategy)) +
    geom_line(linewidth = 0.8) +
    scale_y_log10() +
    scale_colour_manual(values = c(
      setNames("#aaaaaa", sprintf("BUY_&_HOLD  $%.0f (%.0f%%)", bh_final, bh_final - 100)),
      setNames("#6a5acd", sprintf("ZETA_σ.3  $%.0f (%.0f%%)",    z_final,  z_final  - 100)),
      setNames("#00bfff", sprintf("ECLIPSE_σ.4  $%.0f (%.0f%%)", e_final,  e_final  - 100))
    )) +
    labs(
      title = sprintf("\u03b6-Field Backtest  |  ZETA_σ.3 LEADS: $%.0f", z_final),
      x = "Year", y = "PORTFOLIO_VALUE",
      caption = sprintf(
        "MAX_DD — BUY_&_HOLD: %.1f%%  |  ZETA_σ.3: %.1f%%  |  ECLIPSE_σ.4: %.1f%%",
        max_dd(bh_equity), max_dd(eq_z), max_dd(eq_e)
      )
    ) +
    theme(
      plot.background  = element_rect(fill = "black", colour = NA),
      panel.background = element_rect(fill = "black", colour = NA),
      panel.grid.major = element_line(colour = "#333333", linewidth = 0.3),
      panel.grid.minor = element_blank(),
      text             = element_text(colour = "white"),
      axis.text        = element_text(colour = "white"),
      axis.title       = element_text(colour = "white"),
      plot.title       = element_text(colour = "cyan", face = "bold", size = 14),
      plot.caption     = element_text(colour = "grey60", size = 9),
      legend.background = element_rect(fill = "#111111", colour = "grey40"),
      legend.text       = element_text(colour = "white"),
      legend.title      = element_blank(),
      legend.position   = "top"
    )

  ggsave(save_path, plot = p, width = 14, height = 6, dpi = 200, bg = "black")
  cat(sprintf("[✓] Backtest chart saved → %s\n", save_path))
}


# ─────────────────────────────────────────────
# 5. MAIN
# ─────────────────────────────────────────────

cat(strrep("=", 60), "\n")
cat("  \u03b6-Field Model — R Implementation\n")
cat(strrep("=", 60), "\n\n")

cat("[1/2] Generating 3D \u03b6-Field surface …\n")
plot_3d_surface("zeta_field_3d_r.html")

cat("[2/2] Running portfolio backtest …\n")
plot_backtest("zeta_backtest_r.png")

cat("\nDone. Output files:\n")
cat("  • zeta_field_3d_r.html  — interactive 3D field surface\n")
cat("  • zeta_backtest_r.png   — portfolio backtest chart\n")
