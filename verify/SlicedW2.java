// Recompute the published sliced Wasserstein-2 from the exported point clouds.
//
// Sample quality in this repo is one function, sliced_w2 in bench/experiment.py,
// and every cell of both NFE tables came out of it. If it sorted the wrong axis,
// or averaged before taking the square root, or compared unequal counts, the
// numbers would still look plausible and every figure would agree with them,
// because the figures read the same CSV. This reads the two point clouds and
// the projection directions the published run used, implements the metric from
// its definition, and has to land on the published value.
//
//   java verify/SlicedW2.java <repo-root>

import java.io.BufferedReader;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class SlicedW2 {
    static double[][] readMatrix(Path p, int cols) throws IOException {
        List<double[]> rows = new ArrayList<>();
        try (BufferedReader r = Files.newBufferedReader(p)) {
            String line;
            while ((line = r.readLine()) != null) {
                line = line.trim();
                if (line.isEmpty()) continue;
                String[] parts = line.split("\\s+");
                if (parts.length != cols) {
                    throw new IOException(p + ": row has " + parts.length + " values, expected " + cols);
                }
                double[] row = new double[cols];
                for (int i = 0; i < cols; i++) row[i] = Double.parseDouble(parts[i]);
                rows.add(row);
            }
        }
        if (rows.isEmpty()) throw new IOException(p + " is empty");
        return rows.toArray(new double[0][]);
    }

    /** Exact 1D W2 per projection, averaged over directions, then square rooted. */
    static double slicedW2(double[][] a, double[][] b, double[][] dirs) {
        int n = Math.min(a.length, b.length);
        double total = 0.0;
        long count = 0;
        for (double[] d : dirs) {
            double[] pa = project(a, d);
            double[] pb = project(b, d);
            Arrays.sort(pa);
            Arrays.sort(pb);
            for (int i = 0; i < n; i++) {
                double diff = pa[i] - pb[i];
                total += diff * diff;
                count++;
            }
        }
        return Math.sqrt(total / count);
    }

    static double[] project(double[][] x, double[] d) {
        double[] out = new double[x.length];
        for (int i = 0; i < x.length; i++) {
            double acc = 0.0;
            for (int j = 0; j < d.length; j++) acc += x[i][j] * d[j];
            out[i] = acc;
        }
        return out;
    }

    /** The published cell, with columns resolved by name. */
    static double published(Path csv, String dataset, String seed, String model, String nfe)
            throws IOException {
        List<String> lines = Files.readAllLines(csv);
        String[] header = lines.get(0).trim().split(",");
        int iDs = -1, iSeed = -1, iModel = -1, iSampler = -1, iNfe = -1, iW2 = -1;
        for (int i = 0; i < header.length; i++) {
            switch (header[i]) {
                case "dataset": iDs = i; break;
                case "seed": iSeed = i; break;
                case "model": iModel = i; break;
                case "sampler": iSampler = i; break;
                case "nfe": iNfe = i; break;
                case "sliced_w2": iW2 = i; break;
                default: break;
            }
        }
        if (iDs < 0 || iSeed < 0 || iModel < 0 || iSampler < 0 || iNfe < 0 || iW2 < 0) {
            throw new IOException(csv + " is missing one of the columns this reads by name");
        }
        for (String line : lines.subList(1, lines.size())) {
            if (line.isBlank()) continue;
            String[] f = line.trim().split(",");
            if (f.length != header.length) {
                throw new IOException(csv + ": row has " + f.length + " fields, header has "
                        + header.length);
            }
            if (f[iDs].equals(dataset) && f[iSeed].equals(seed) && f[iModel].equals(model)
                    && f[iSampler].equals("euler") && f[iNfe].equals(nfe)) {
                return Double.parseDouble(f[iW2]);
            }
        }
        throw new IOException("no " + dataset + "/" + seed + "/" + model + " row at NFE " + nfe);
    }

    public static void main(String[] args) throws IOException {
        Path root = Path.of(args.length > 0 ? args[0] : ".");
        Path golden = root.resolve("verify/golden");
        double[][] ref = readMatrix(golden.resolve("w2-reference.txt"), 2);
        double[][] dirs = readMatrix(golden.resolve("w2-directions.txt"), 2);
        double tol = 1e-5;
        boolean bad = false;

        for (String model : new String[] {"2-rectified", "diffusion-vp"}) {
            double[][] gen = readMatrix(
                    golden.resolve("w2-samples-" + model + "-nfe1.txt"), 2);
            double got = slicedW2(gen, ref, dirs);
            double want = published(root.resolve("results/nfe-quality.csv"),
                    "8gaussians", "0", model, "1");
            double rel = Math.abs(got - want) / Math.max(Math.abs(want), 1e-12);
            System.out.printf("  %-13s Java %.10f  published %.10f  relative %.2e%n",
                    model, got, want, rel);
            if (!(rel <= tol)) {
                System.out.printf("    DISAGREES, tolerance is %.1e%n", tol);
                bad = true;
            }
        }
        System.exit(bad ? 1 : 0);
    }
}
