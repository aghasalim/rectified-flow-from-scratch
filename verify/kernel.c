/* Independent C reimplementation of the velocity MLP, the Euler sampler and
 * the straightness metric, checked against the published results/straightness.csv.
 *
 * fm/models.py, fm/samplers.py and fm/straightness.py are one implementation,
 * and every S in the README came out of it. If the time embedding used the
 * wrong frequency, or the sampler integrated with an off-by-one t, or the
 * metric compared against a data sample instead of the trajectory endpoint,
 * nothing in the repo would notice: the figures read the same CSV the tables do.
 * This reads the exported weights and the exact noise batch the published run
 * used, integrates the ODE itself, and has to land on the same S.
 *
 * Weight matrices are looked up BY NAME, so a reordered state dict fails here
 * rather than quietly multiplying the wrong tensor.
 *
 *   cc -std=c99 -O2 -o kernel verify/kernel.c -lm
 *   ./kernel <repo-root> <model-name>
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define STEPS 100
#define TDIM 64
#define HID 256
#define IN (2 + TDIM)
#define MAXT 32

typedef struct { char name[64]; int rows, cols; float *v; } Tensor;

static Tensor tensors[MAXT];
static int n_tensors = 0;

static const Tensor *get(const char *name)
{
    for (int i = 0; i < n_tensors; i++)
        if (strcmp(tensors[i].name, name) == 0) return &tensors[i];
    fprintf(stderr, "kernel: no tensor named %s in the exported weights\n", name);
    exit(1);
}

static void load_weights(const char *path)
{
    FILE *f = fopen(path, "r");
    if (!f) { fprintf(stderr, "kernel: cannot open %s\n", path); exit(1); }
    char name[64];
    int rows, cols;
    while (fscanf(f, "%63s %d %d", name, &rows, &cols) == 3) {
        if (n_tensors == MAXT) { fprintf(stderr, "kernel: too many tensors\n"); exit(1); }
        Tensor *t = &tensors[n_tensors++];
        snprintf(t->name, sizeof t->name, "%s", name);
        t->rows = rows; t->cols = cols;
        t->v = malloc((size_t)rows * cols * sizeof *t->v);
        if (!t->v) { fprintf(stderr, "kernel: out of memory\n"); exit(1); }
        for (long i = 0; i < (long)rows * cols; i++)
            if (fscanf(f, "%f", &t->v[i]) != 1) {
                fprintf(stderr, "kernel: %s is short at element %ld\n", name, i);
                exit(1);
            }
    }
    fclose(f);
}

static double *load_matrix(const char *path, int cols, int *rows_out)
{
    FILE *f = fopen(path, "r");
    if (!f) { fprintf(stderr, "kernel: cannot open %s\n", path); exit(1); }
    size_t cap = 1024, n = 0;
    double *v = malloc(cap * sizeof *v), x;
    while (fscanf(f, "%lf", &x) == 1) {
        if (n == cap) { cap *= 2; v = realloc(v, cap * sizeof *v); }
        v[n++] = x;
    }
    fclose(f);
    if (n == 0 || n % (size_t)cols) {
        fprintf(stderr, "kernel: %s holds %zu values, not a multiple of %d\n", path, n, cols);
        exit(1);
    }
    *rows_out = (int)(n / cols);
    return v;
}

static double silu(double z) { return z / (1.0 + exp(-z)); }

/* y = W x + b, with W stored row major as (out, in). */
static void affine(const Tensor *W, const Tensor *b, const double *x, double *y)
{
    for (int o = 0; o < W->rows; o++) {
        const float *row = W->v + (size_t)o * W->cols;
        double acc = b->v[o];
        for (int i = 0; i < W->cols; i++) acc += (double)row[i] * x[i];
        y[o] = acc;
    }
}

/* The time embedding depends only on t, which is constant across the batch in
 * every sampler here, so it is computed once per step rather than per point. */
static void time_embedding(double t, double *out)
{
    double raw[TDIM], h[TDIM];
    int half = TDIM / 2;
    for (int j = 0; j < half; j++) {
        double freq = exp(-log(10000.0) * (double)j / half);
        double ang = t * freq * 1000.0;
        raw[j] = sin(ang);
        raw[half + j] = cos(ang);
    }
    affine(get("time.mlp.0.weight"), get("time.mlp.0.bias"), raw, h);
    for (int j = 0; j < TDIM; j++) h[j] = silu(h[j]);
    affine(get("time.mlp.2.weight"), get("time.mlp.2.bias"), h, out);
}

static void velocity(const double *x, const double *temb, double *v)
{
    double in[IN], a[HID], b[HID];
    in[0] = x[0]; in[1] = x[1];
    memcpy(in + 2, temb, TDIM * sizeof *temb);

    affine(get("net.0.weight"), get("net.0.bias"), in, a);
    for (int i = 0; i < HID; i++) a[i] = silu(a[i]);
    const char *ws[3] = {"net.2.weight", "net.4.weight", "net.6.weight"};
    const char *bs[3] = {"net.2.bias", "net.4.bias", "net.6.bias"};
    for (int l = 0; l < 3; l++) {
        affine(get(ws[l]), get(bs[l]), a, b);
        for (int i = 0; i < HID; i++) a[i] = silu(b[i]);
    }
    affine(get("net.8.weight"), get("net.8.bias"), a, v);
}

static int cmp_double(const void *p, const void *q)
{
    double a = *(const double *)p, b = *(const double *)q;
    return (a > b) - (a < b);
}

/* torch.quantile's default is linear interpolation between order statistics. */
static double quantile(double *sorted, int n, double q)
{
    double pos = q * (n - 1);
    int lo = (int)floor(pos);
    int hi = lo + 1 < n ? lo + 1 : n - 1;
    double frac = pos - lo;
    return sorted[lo] * (1.0 - frac) + sorted[hi] * frac;
}

/* Published row from results/straightness.csv, columns resolved by name. */
static int published(const char *root, const char *model, double *S,
                     double *ratio_mean, double *ratio_p90)
{
    char path[1024];
    snprintf(path, sizeof path, "%s/results/straightness.csv", root);
    FILE *f = fopen(path, "r");
    if (!f) { fprintf(stderr, "kernel: cannot open %s\n", path); return 1; }
    char line[4096];
    if (!fgets(line, sizeof line, f)) { fclose(f); return 1; }
    line[strcspn(line, "\r\n")] = 0;
    int idx[32], ncol = 0;
    const char *want[] = {"dataset", "seed", "model", "straightness_S",
                          "path_length_ratio_mean", "path_length_ratio_p90"};
    int found[6] = {0};
    for (char *tok = strtok(line, ","); tok; tok = strtok(NULL, ",")) {
        for (int k = 0; k < 6; k++)
            if (strcmp(tok, want[k]) == 0) { idx[k] = ncol; found[k] = 1; }
        ncol++;
    }
    for (int k = 0; k < 6; k++)
        if (!found[k]) {
            fprintf(stderr, "kernel: straightness.csv has no column %s\n", want[k]);
            fclose(f); return 1;
        }
    while (fgets(line, sizeof line, f)) {
        line[strcspn(line, "\r\n")] = 0;
        char *fields[64]; int n = 0;
        for (char *tok = strtok(line, ","); tok && n < 64; tok = strtok(NULL, ",")) fields[n++] = tok;
        if (n != ncol) {
            fprintf(stderr, "kernel: straightness.csv has a row with %d fields, header has %d\n",
                    n, ncol);
            fclose(f); return 1;
        }
        if (strcmp(fields[idx[0]], "8gaussians") == 0 && strcmp(fields[idx[1]], "0") == 0
            && strcmp(fields[idx[2]], model) == 0) {
            *S = atof(fields[idx[3]]);
            *ratio_mean = atof(fields[idx[4]]);
            *ratio_p90 = atof(fields[idx[5]]);
            fclose(f);
            return 0;
        }
    }
    fprintf(stderr, "kernel: no 8gaussians/seed 0/%s row in straightness.csv\n", model);
    fclose(f);
    return 1;
}

int main(int argc, char **argv)
{
    if (argc != 3) { fprintf(stderr, "usage: kernel <repo-root> <model>\n"); return 2; }
    const char *root = argv[1], *model = argv[2];
    char path[1024];

    snprintf(path, sizeof path, "%s/verify/golden/weights-%s.txt", root, model);
    load_weights(path);
    snprintf(path, sizeof path, "%s/verify/golden/x0-straightness.txt", root);
    int nb = 0;
    double *x0 = load_matrix(path, 2, &nb);

    double S_pub, mean_pub, p90_pub;
    if (published(root, model, &S_pub, &mean_pub, &p90_pub)) return 1;

    double dt = 1.0 / STEPS;
    static double temb[STEPS][TDIM];
    for (int k = 0; k < STEPS; k++) time_embedding(k * dt, temb[k]);
    double *ratio = malloc((size_t)nb * sizeof *ratio);
    double s_total = 0.0;

    for (int b = 0; b < nb; b++) {
        double x[2] = {x0[2 * b], x0[2 * b + 1]};
        double sum_v[2] = {0, 0}, sum_v2 = 0.0, seg = 0.0;
        for (int i = 0; i < STEPS; i++) {
            double v[2];
            velocity(x, temb[i], v);
            /* || (x1 - x0) - v ||^2 expanded, so the integral needs one pass. */
            sum_v[0] += v[0] * dt; sum_v[1] += v[1] * dt;
            sum_v2 += (v[0] * v[0] + v[1] * v[1]) * dt;
            x[0] += dt * v[0]; x[1] += dt * v[1];
            seg += dt * sqrt(v[0] * v[0] + v[1] * v[1]);
        }
        double d0 = x[0] - x0[2 * b], d1 = x[1] - x0[2 * b + 1];
        double chord = sqrt(d0 * d0 + d1 * d1);
        s_total += (d0 * d0 + d1 * d1) - 2.0 * (d0 * sum_v[0] + d1 * sum_v[1]) + sum_v2;
        ratio[b] = seg / (chord < 1e-12 ? 1e-12 : chord);
    }

    double S = s_total / nb, mean = 0.0;
    for (int b = 0; b < nb; b++) mean += ratio[b];
    mean /= nb;
    qsort(ratio, nb, sizeof *ratio, cmp_double);
    double p90 = quantile(ratio, nb, 0.9);

    struct { const char *name; double got, want, tol; } checks[] = {
        {"straightness_S",         S,    S_pub,    1e-5},
        {"path_length_ratio_mean", mean, mean_pub, 1e-5},
        {"path_length_ratio_p90",  p90,  p90_pub,  1e-5},
    };
    int bad = 0;
    printf("C kernel, %s, %d trajectories x %d Euler steps\n", model, nb, STEPS);
    for (size_t k = 0; k < sizeof checks / sizeof *checks; k++) {
        double rel = fabs(checks[k].got - checks[k].want) / fmax(fabs(checks[k].want), 1e-12);
        printf("  %-24s C %.10g  published %.10g  relative %.2e\n",
               checks[k].name, checks[k].got, checks[k].want, rel);
        if (!(rel <= checks[k].tol)) {
            printf("    DISAGREES, tolerance is %.1e\n", checks[k].tol);
            bad = 1;
        }
    }
    return bad;
}
