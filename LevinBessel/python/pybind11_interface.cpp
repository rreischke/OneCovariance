#include <vector>
#include <iostream>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <pybind11/iostream.h>

#include "./../src/levin.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>

namespace py = pybind11;
using namespace pybind11::literals;


typedef py::array_t<double, py::array::c_style | py::array::forcecast> darray;

// Piecewise cubic Hermite interpolant (in u = ln x) through (lx, Y[:, c]) with
// derivatives YP[:, c], continued linearly beyond lx with slope0 (below) and
// slope (above), evaluated at le; exp of it for the columns with positive[c]
// (interpolation of ln y). With derivative, also the derivatives with
// respect to u from the right and from the left (of exp(...) if positive):
// they differ only at the nodes of lx. This is the evaluation part of
// CovELLSpace._integrand_at (which prepares the arguments), done in parallel
// over the evaluation points.
py::tuple evaluate_hermite(darray lx_in, darray Y_in, darray YP_in, darray slope_in, darray slope0_in,
                           py::array_t<bool, py::array::c_style | py::array::forcecast> positive_in,
                           darray le_in, bool derivative, int n_threads)
{
     const ssize_t n = lx_in.shape(0), nc = Y_in.shape(1), m = le_in.shape(0);
     if (n < 2 || Y_in.shape(0) != n || YP_in.shape(0) != n || YP_in.shape(1) != nc
         || slope_in.shape(0) != nc || slope0_in.shape(0) != nc || positive_in.shape(0) != nc)
     {
          throw std::invalid_argument("evaluate_hermite: inconsistent shapes");
     }
     darray values({m, nc});
     darray d_right(derivative ? std::vector<ssize_t>{m, nc} : std::vector<ssize_t>{0, nc});
     darray d_left(derivative ? std::vector<ssize_t>{m, nc} : std::vector<ssize_t>{0, nc});
     const double *lx = lx_in.data(), *Y = Y_in.data(), *YP = YP_in.data(), *le = le_in.data();
     const double *slope = slope_in.data(), *slope0 = slope0_in.data();
     const bool *positive = positive_in.data();
     double *v = values.mutable_data();
     double *dr = derivative ? d_right.mutable_data() : nullptr;
     double *dl = derivative ? d_left.mutable_data() : nullptr;
     {
          py::gil_scoped_release release;
#pragma omp parallel for num_threads(std::max(n_threads, 1)) schedule(static)
          for (ssize_t e = 0; e < m; e++)
          {
               const double u = le[e];
               double *ve = v + e*nc;
               // values (the interval to the right at the nodes, as numpy.searchsorted(side='right'))
               const bool above = u >= lx[n - 1], below = u < lx[0];
               ssize_t i = 0;
               double h = 1.0, t = 0.0;
               if (!above && !below)
               {
                    i = std::min<ssize_t>(std::max<ssize_t>(std::upper_bound(lx, lx + n, u) - lx - 1, 0), n - 2);
                    h = lx[i + 1] - lx[i];
                    t = (u - lx[i])/h;
               }
               const double *Yi = Y + i*nc, *Yj = Y + (i + 1)*nc, *Pi = YP + i*nc, *Pj = YP + (i + 1)*nc;
               if (above)
               {
                    for (ssize_t c = 0; c < nc; c++)
                         ve[c] = Y[(n - 1)*nc + c] + slope[c]*(u - lx[n - 1]);
               }
               else if (below)
               {
                    for (ssize_t c = 0; c < nc; c++)
                         ve[c] = Y[c] + slope0[c]*(u - lx[0]);
               }
               else
               {
                    const double a0 = 2*t*t*t - 3*t*t + 1, a1 = 3*t*t - 2*t*t*t;
                    const double b0 = h*(t*t*t - 2*t*t + t), b1 = h*(t*t*t - t*t);
                    for (ssize_t c = 0; c < nc; c++)
                         ve[c] = (a0*Yi[c] + a1*Yj[c]) + (b0*Pi[c] + b1*Pj[c]);
               }
               for (ssize_t c = 0; c < nc; c++)
                    if (positive[c])
                         ve[c] = std::exp(ve[c]);
               if (!derivative)
                    continue;
               // derivative from the right
               double *dre = dr + e*nc;
               if (above)
               {
                    for (ssize_t c = 0; c < nc; c++)
                         dre[c] = slope[c];
               }
               else if (below)
               {
                    for (ssize_t c = 0; c < nc; c++)
                         dre[c] = slope0[c];
               }
               else
               {
                    const double a0 = 6*(t*t - t)/h, a1 = 6*(t - t*t)/h;
                    const double b0 = 3*t*t - 4*t + 1, b1 = 3*t*t - 2*t;
                    for (ssize_t c = 0; c < nc; c++)
                         dre[c] = (a0*Yi[c] + a1*Yj[c]) + (b0*Pi[c] + b1*Pj[c]);
               }
               for (ssize_t c = 0; c < nc; c++)
                    if (positive[c])
                         dre[c] *= ve[c];
               // derivative from the left: differs only at the nodes
               double *dle = dl + e*nc;
               if (!std::binary_search(lx, lx + n, u))
               {
                    std::copy(dre, dre + nc, dle);
                    continue;
               }
               const bool above_l = u > lx[n - 1], below_l = u <= lx[0];
               if (above_l)
               {
                    for (ssize_t c = 0; c < nc; c++)
                         dle[c] = slope[c];
               }
               else if (below_l)
               {
                    for (ssize_t c = 0; c < nc; c++)
                         dle[c] = slope0[c];
               }
               else
               {
                    const ssize_t k = std::min<ssize_t>(std::max<ssize_t>(std::lower_bound(lx, lx + n, u) - lx - 1, 0), n - 2);
                    const double hk = lx[k + 1] - lx[k], tk = (u - lx[k])/hk;
                    const double a0 = 6*(tk*tk - tk)/hk, a1 = 6*(tk - tk*tk)/hk;
                    const double b0 = 3*tk*tk - 4*tk + 1, b1 = 3*tk*tk - 2*tk;
                    const double *Yk = Y + k*nc, *Yl = Y + (k + 1)*nc, *Pk = YP + k*nc, *Pl = YP + (k + 1)*nc;
                    for (ssize_t c = 0; c < nc; c++)
                         dle[c] = (a0*Yk[c] + a1*Yl[c]) + (b0*Pk[c] + b1*Pl[c]);
               }
               for (ssize_t c = 0; c < nc; c++)
                    if (positive[c])
                         dle[c] *= ve[c];
          }
     }
     if (!derivative)
          return py::make_tuple(values);
     return py::make_tuple(values, d_right, d_left);
}

typedef py::array_t<int64_t, py::array::c_style | py::array::forcecast> iarray;

// tile of pairs of weights (GAUSS_TILE x GAUSS_TILE) handled by one task and
// number of segments whose moments are stored at a time
constexpr ssize_t GAUSS_TILE = 16;
constexpr ssize_t GAUSS_SEGMENTS = 256;
static_assert(GAUSS_TILE % 8 == 0, "register blocks of 8 pairs");

// vectors of 4 doubles (GCC/clang vector extensions: NEON, SSE2 or AVX), and
// the rows of the register blocks of the moments (fewer with the 16 registers
// of SSE2, which would otherwise spill)
typedef double v4d __attribute__((vector_size(32)));
static inline v4d load4(const double *p)
{
     v4d v;
     std::memcpy(&v, p, sizeof(v));
     return v;
}
static inline void store4(double *p, v4d v)
{
     std::memcpy(p, &v, sizeof(v));
}
#if defined(__aarch64__) || defined(__AVX__)
constexpr ssize_t GAUSS_ROWS = 4;
#else
constexpr ssize_t GAUSS_ROWS = 2;
#endif

// The Gaussian projection of CovELLSpace._project_gauss_quadrature_keys:
//   result[row, c] = sum_j w_j W_a(x_j) W_b(x_j) p_c(x_j)
// for the pairs of weights (a, b) = (first[i], second[l]) with
// row = row_of[i*len(second) + l] >= 0, where p_c is the piecewise cubic
// Hermite interpolant of column c of the integrand on the sampling grid. The
// nodes x_j are grouped into segments [starts[s], stops[s]) within the grid
// interval k[s]; there p_c(x_j) = coefficients[0, j] values[k, c]
// + coefficients[1, j] values[k + 1, c] + coefficients[2, j] d_right[k, c]
// + coefficients[3, j] d_left[k + 1, c]. The weights on the nodes are
// W_lo[a] (first node), W[a, j - 1] (interior nodes) and W_hi[a] (last node);
// W only needs unit stride along the nodes (a slice of the registered
// weights). For every segment and pair the four moments
// sum_j coefficients[q, j] w_j W_a(x_j) W_b(x_j) are summed first and then
// contracted with the integrand, in parallel over tiles of pairs and chunks of
// segments (each task owns its results, chunks are summed at the end).
darray gauss_hermite_projection(py::array_t<double, py::array::forcecast> W_in, darray W_lo_in, darray W_hi_in,
                                darray w_in, iarray first_in, iarray second_in, iarray row_of_in, ssize_t n_rows,
                                darray coefficients_in, iarray starts_in, iarray stops_in, iarray k_in,
                                darray values_in, darray d_right_in, darray d_left_in, int n_threads)
{
     const ssize_t n_x = w_in.shape(0), n_first = first_in.shape(0), n_second = second_in.shape(0);
     const ssize_t n_seg = starts_in.shape(0), n_grid = values_in.shape(0), nc = values_in.shape(1);
     if (W_in.ndim() != 2 || W_in.shape(1) != n_x - 2 || (n_x > 2 && W_in.strides(1) != sizeof(double))
         || W_lo_in.shape(0) != W_in.shape(0) || W_hi_in.shape(0) != W_in.shape(0)
         || row_of_in.shape(0) != n_first*n_second || coefficients_in.shape(0) != 4 || coefficients_in.shape(1) != n_x
         || stops_in.shape(0) != n_seg || k_in.shape(0) != n_seg
         || d_right_in.shape(0) != n_grid || d_left_in.shape(0) != n_grid || d_right_in.shape(1) != nc
         || d_left_in.shape(1) != nc)
     {
          throw std::invalid_argument("gauss_hermite_projection: inconsistent shapes");
     }
     const ssize_t n_modes = W_in.shape(0), W_row = W_in.strides(0)/ssize_t(sizeof(double));
     const double *W = W_in.data(), *W_lo = W_lo_in.data(), *W_hi = W_hi_in.data(), *w = w_in.data();
     const int64_t *first = first_in.data(), *second = second_in.data(), *row_of = row_of_in.data();
     const int64_t *starts = starts_in.data(), *stops = stops_in.data(), *k = k_in.data();
     const double *coefficients = coefficients_in.data();
     const double *values = values_in.data(), *d_right = d_right_in.data(), *d_left = d_left_in.data();
     for (ssize_t i = 0; i < n_first; i++)
          if (first[i] < 0 || first[i] >= n_modes)
               throw std::invalid_argument("gauss_hermite_projection: mode out of range");
     for (ssize_t i = 0; i < n_second; i++)
          if (second[i] < 0 || second[i] >= n_modes)
               throw std::invalid_argument("gauss_hermite_projection: mode out of range");
     ssize_t max_length = 1;
     for (ssize_t s = 0; s < n_seg; s++)
     {
          if (starts[s] < 0 || stops[s] > n_x || stops[s] < starts[s] || k[s] < 0 || k[s] + 1 >= n_grid)
               throw std::invalid_argument("gauss_hermite_projection: segment out of range");
          max_length = std::max<ssize_t>(max_length, stops[s] - starts[s]);
     }
     for (ssize_t i = 0; i < n_first*n_second; i++)
          if (row_of[i] >= n_rows)
               throw std::invalid_argument("gauss_hermite_projection: row out of range");
     darray result({n_rows, nc});
     double *out = result.mutable_data();
     std::fill(out, out + n_rows*nc, 0.0);
     // the tiles with at least one requested pair
     const ssize_t tiles_first = (n_first + GAUSS_TILE - 1)/GAUSS_TILE, tiles_second = (n_second + GAUSS_TILE - 1)/GAUSS_TILE;
     std::vector<ssize_t> tiles;
     for (ssize_t ti = 0; ti < tiles_first; ti++)
          for (ssize_t tl = 0; tl < tiles_second; tl++)
          {
               bool active = false;
               for (ssize_t i = ti*GAUSS_TILE; i < std::min(n_first, (ti + 1)*GAUSS_TILE) && !active; i++)
                    for (ssize_t l = tl*GAUSS_TILE; l < std::min(n_second, (tl + 1)*GAUSS_TILE); l++)
                         if (row_of[i*n_second + l] >= 0)
                         {
                              active = true;
                              break;
                         }
               if (active)
                    tiles.push_back(ti*tiles_second + tl);
          }
     if (tiles.empty() || nc == 0)
          return result;
     const int threads = std::max(n_threads, 1);
     // chunks of segments, so that there are a few tasks per thread (at least 16
     // segments per chunk, their results within 2^24 elements)
     ssize_t n_chunks = (4*threads + ssize_t(tiles.size()) - 1)/ssize_t(tiles.size());
     n_chunks = std::min<ssize_t>(n_chunks, std::max<ssize_t>(1, n_seg/16));
     n_chunks = std::max<ssize_t>(1, std::min<ssize_t>(n_chunks, 1 + (ssize_t(1) << 24)/std::max<ssize_t>(n_rows*nc, 1)));
     const ssize_t chunk_length = (n_seg + n_chunks - 1)/n_chunks;
     // the results of the chunks after the first (summed at the end)
     std::vector<double> partial(n_chunks > 1 ? (n_chunks - 1)*n_rows*nc : 0, 0.0);
     const ssize_t n_tasks = ssize_t(tiles.size())*n_chunks;
     {
          py::gil_scoped_release release;
#pragma omp parallel num_threads(threads)
          {
               constexpr ssize_t T = GAUSS_TILE, P = GAUSS_TILE*GAUSS_TILE;
               std::vector<double> A((4*T + 1)*max_length), B(max_length*T), G(4*P*GAUSS_SEGMENTS), V(4*8*GAUSS_SEGMENTS);
               // the results of the task: [column block][pair][8 columns]
               std::vector<double> acc(P*8*((nc + 7)/8));
               // the weights of mode a on the nodes [start, start + length)
               auto weights_on = [&](ssize_t a, ssize_t start, ssize_t length, double *row)
               {
                    const ssize_t lo = std::max<ssize_t>(start, 1), hi = std::min<ssize_t>(start + length, n_x - 1);
                    if (hi > lo)
                         std::copy(W + a*W_row + lo - 1, W + a*W_row + hi - 1, row + lo - start);
                    if (start == 0)
                         row[0] = W_lo[a];
                    if (start + length == n_x)
                         row[length - 1] = W_hi[a];
               };
#pragma omp for schedule(dynamic, 1)
               for (ssize_t task = 0; task < n_tasks; task++)
               {
                    const ssize_t tile = tiles[task/n_chunks], chunk = task % n_chunks;
                    const ssize_t i0 = (tile/tiles_second)*T, l0 = (tile % tiles_second)*T;
                    const ssize_t n_i = std::min(T, n_first - i0), n_l = std::min(T, n_second - l0);
                    const ssize_t s0 = chunk*chunk_length, s1 = std::min(n_seg, s0 + chunk_length);
                    if (s0 >= s1)
                         continue;
                    std::fill(acc.begin(), acc.end(), 0.0);
                    for (ssize_t b0 = s0; b0 < s1; b0 += GAUSS_SEGMENTS)
                    {
                         const ssize_t b1 = std::min(s1, b0 + GAUSS_SEGMENTS);
                         // moments G[s][q][pair] of the segments of the sub-chunk
                         for (ssize_t s = b0; s < b1; s++)
                         {
                              const ssize_t start = starts[s], length = stops[s] - start;
                              double *Gs = G.data() + (s - b0)*4*P;
                              if (length == 0)
                              {
                                   std::fill(Gs, Gs + 4*P, 0.0);
                                   continue;
                              }
                              // A[q][i][j] = coefficients[q, j] W_a(x_j), B[j][l] = w_j W_b(x_j) (zero beyond the tile)
                              const double *coefficients_s = coefficients + start;
                              for (ssize_t i = 0; i < T; i++)
                              {
                                   double *row = A.data() + 4*T*max_length;
                                   if (i < n_i)
                                        weights_on(first[i0 + i], start, length, row);
                                   else
                                        std::fill(row, row + length, 0.0);
                                   for (ssize_t q = 0; q < 4; q++)
                                   {
                                        double *Aq = A.data() + (q*T + i)*max_length;
                                        const double *cq = coefficients_s + q*n_x;
                                        for (ssize_t j = 0; j < length; j++)
                                             Aq[j] = cq[j]*row[j];
                                   }
                              }
                              for (ssize_t l = 0; l < T; l++)
                              {
                                   double *row = A.data() + 4*T*max_length;
                                   if (l < n_l)
                                        weights_on(second[l0 + l], start, length, row);
                                   else
                                        std::fill(row, row + length, 0.0);
                                   for (ssize_t j = 0; j < length; j++)
                                        B[j*T + l] = w[start + j]*row[j];
                              }
                              // register blocks of GAUSS_ROWS rows (q, i) and 8 pairs l
                              for (ssize_t qi = 0; qi < 4*T; qi += GAUSS_ROWS)
                              {
                                   const double *A0 = A.data() + qi*max_length;
                                   for (ssize_t lh = 0; lh < T; lh += 8)
                                   {
                                        v4d g[GAUSS_ROWS][2] = {};
                                        for (ssize_t j = 0; j < length; j++)
                                        {
                                             const v4d b0 = load4(B.data() + j*T + lh), b1 = load4(B.data() + j*T + lh + 4);
                                             for (ssize_t r = 0; r < GAUSS_ROWS; r++)
                                             {
                                                  const double a = A0[r*max_length + j];
                                                  g[r][0] += a*b0;
                                                  g[r][1] += a*b1;
                                             }
                                        }
                                        // G[s][q][i*T + l] with qi + r = q*T + i
                                        for (ssize_t r = 0; r < GAUSS_ROWS; r++)
                                        {
                                             store4(Gs + (qi + r)*T + lh, g[r][0]);
                                             store4(Gs + (qi + r)*T + lh + 4, g[r][1]);
                                        }
                                   }
                              }
                         }
                         // contraction with the integrand: register blocks of GAUSS_ROWS pairs and
                         // 8 columns, accumulated over the segments of the sub-chunk
                         for (ssize_t c0 = 0; c0 < nc; c0 += 8)
                         {
                              const ssize_t n_c = std::min<ssize_t>(8, nc - c0);
                              // V[s][q][c] of the sub-chunk (zero beyond the columns)
                              for (ssize_t s = b0; s < b1; s++)
                              {
                                   const double *rows[4] = {values + k[s]*nc + c0, values + (k[s] + 1)*nc + c0,
                                                            d_right + k[s]*nc + c0, d_left + (k[s] + 1)*nc + c0};
                                   for (ssize_t q = 0; q < 4; q++)
                                   {
                                        double *Vq = V.data() + ((s - b0)*4 + q)*8;
                                        for (ssize_t c = 0; c < 8; c++)
                                             Vq[c] = c < n_c ? rows[q][c] : 0.0;
                                   }
                              }
                              double *acc_c = acc.data() + (c0/8)*P*8;
                              for (ssize_t p = 0; p < P; p += GAUSS_ROWS)
                              {
                                   v4d e[GAUSS_ROWS][2];
                                   for (ssize_t r = 0; r < GAUSS_ROWS; r++)
                                   {
                                        e[r][0] = load4(acc_c + (p + r)*8);
                                        e[r][1] = load4(acc_c + (p + r)*8 + 4);
                                   }
                                   for (ssize_t sq = 0; sq < 4*(b1 - b0); sq++)
                                   {
                                        // G[s][q][p] with sq = 4 (s - b0) + q
                                        const double *g = G.data() + sq*P + p;
                                        const v4d v0 = load4(V.data() + sq*8), v1 = load4(V.data() + sq*8 + 4);
                                        for (ssize_t r = 0; r < GAUSS_ROWS; r++)
                                        {
                                             e[r][0] += g[r]*v0;
                                             e[r][1] += g[r]*v1;
                                        }
                                   }
                                   for (ssize_t r = 0; r < GAUSS_ROWS; r++)
                                   {
                                        store4(acc_c + (p + r)*8, e[r][0]);
                                        store4(acc_c + (p + r)*8 + 4, e[r][1]);
                                   }
                              }
                         }
                    }
                    double *target = chunk == 0 ? out : partial.data() + (chunk - 1)*n_rows*nc;
                    for (ssize_t i = 0; i < n_i; i++)
                         for (ssize_t l = 0; l < n_l; l++)
                         {
                              const int64_t row = row_of[(i0 + i)*n_second + l0 + l];
                              if (row < 0)
                                   continue;
                              for (ssize_t c0 = 0; c0 < nc; c0 += 8)
                              {
                                   const double *e = acc.data() + ((c0/8)*P + i*T + l)*8;
                                   std::copy(e, e + std::min<ssize_t>(8, nc - c0), target + row*nc + c0);
                              }
                         }
               }
               if (n_chunks > 1)
               {
#pragma omp for schedule(static)
                    for (ssize_t e = 0; e < n_rows*nc; e++)
                         for (ssize_t chunk = 1; chunk < n_chunks; chunk++)
                              out[e] += partial[(chunk - 1)*n_rows*nc + e];
               }
          }
     }
     return result;
}

PYBIND11_MODULE(levin, m)
{
     m.doc() = "Compute integrals with Levin's method.";
     m.def("evaluate_hermite", &evaluate_hermite,
           "lx"_a, "Y"_a, "YP"_a, "slope"_a, "slope0"_a, "positive"_a, "le"_a, "derivative"_a, "n_threads"_a);
     m.def("gauss_hermite_projection", &gauss_hermite_projection,
           "W"_a, "W_lo"_a, "W_hi"_a, "w"_a, "first"_a, "second"_a, "row_of"_a, "n_rows"_a,
           "coefficients"_a, "starts"_a, "stops"_a, "k"_a, "values"_a, "d_right"_a, "d_left"_a, "n_threads"_a);
     py::class_<Levin>(m, "Levin")
         .def(py::init<uint, uint, uint, double, uint, uint>(),
              "type1"_a, "col1"_a, "nsub1"_a, "relative_tol1"_a, "n_split_rs1"_a, "Nthread"_a) // Keyword arguments
         .def("update_Levin", &Levin::update_Levin,
              "type1"_a, "col1"_a, "nsub1"_a, "relative_tol1"_a, "converged1"_a,
              py::call_guard<py::gil_scoped_release>())
         .def("init_integral", &Levin::init_integral,
              "x"_a, "integrand"_a, "logx1"_a, "logy1"_a,
              py::call_guard<py::gil_scoped_release>())
         .def("init_w_ell", &Levin::init_w_ell,
              "ell"_a, "w_ell"_a,
              py::call_guard<py::gil_scoped_release>())
         .def("get_w_ell", &Levin::get_w_ell,
              "ell"_a, "mode"_a,
              py::call_guard<py::gil_scoped_release>())
         .def("single_bessel", &Levin::single_bessel,
              "k"_a, "ell"_a, "a"_a, "b"_a,
              py::call_guard<py::gil_scoped_release>())
         .def("double_bessel", &Levin::double_bessel,
              "k1"_a, "k2"_a, "ell_1"_a, "ell_2"_a, "a"_a, "b"_a,
              py::call_guard<py::gil_scoped_release>())
         .def("double_bessel_many_args", &Levin::double_bessel_many_args,
              "k1"_a, "k2"_a, "ell_1"_a, "ell_2"_a, "a"_a, "b"_a,
              py::call_guard<py::gil_scoped_release>())
         .def("get_integrand",&Levin::get_integrand,
              "x"_a, "j"_a,
              py::call_guard<py::gil_scoped_release>())
         .def("set_kb_levin_min", &Levin::set_kb_levin_min, "kb"_a)
         .def("single_bessel_many_args", &Levin::single_bessel_many_args,
              "k"_a, "ell"_a, "a"_a, "b"_a,
              py::call_guard<py::gil_scoped_release>())
         .def("single_bessel_many_args_diagonal", &Levin::single_bessel_many_args_diagonal,
               "k"_a, "ell"_a, "a"_a, "b"_a,
               py::call_guard<py::gil_scoped_release>())
         .def("cquad_integrate", &Levin::cquad_integrate,
              "limits"_a,
              py::call_guard<py::gil_scoped_release>())
         .def("cquad_integrate_single_well", &Levin::cquad_integrate_single_well,
              "limits"_a, "m_mode"_a,
              py::call_guard<py::gil_scoped_release>())
         .def("cquad_integrate_single_well_subset", &Levin::cquad_integrate_single_well_subset,
              "limits"_a, "m_modes"_a, "indices"_a,
              py::call_guard<py::gil_scoped_release>())
         .def("cquad_integrate_double_well", &Levin::cquad_integrate_double_well,
              "limits"_a, "m_mode"_a, "n_mode"_a,
              py::call_guard<py::gil_scoped_release>())
         .def("cquad_integrate_double_well_subset", &Levin::cquad_integrate_double_well_subset,
              "limits"_a, "m_modes"_a, "n_modes"_a, "indices"_a,
              py::call_guard<py::gil_scoped_release>());
}