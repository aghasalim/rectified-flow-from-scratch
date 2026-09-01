// Structural validation of everything under results/, plus a second
// independent recomputation of the two published median tables.
//
// The CSVs in results/ are the evidence for every number in the README, and
// nothing checked that they are well formed. A truncated write, a column that
// drifted, a NaN out of a division, or a seed that silently went missing so a
// "median of 3 seeds" became a median of two would all be invisible, because
// the figures and the tables read the same file. This walks every tracked
// results file and then recomputes the medians, which verify/verify.sh diffs
// against the SQL and JavaScript versions.
//
//	cd verify/gocheck && go run . -root ..
package main

import (
	"encoding/csv"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
)

var (
	wantSeeds    = []string{"0", "1", "2"}
	wantModels   = []string{"1-rectified", "2-rectified", "diffusion-vp"}
	wantData     = []string{"8gaussians", "moons"}
	publishedNFE = []int{1, 2, 4, 8, 128}
)

type table struct {
	path   string
	header []string
	rows   [][]string
}

func (t *table) col(name string) int {
	for i, h := range t.header {
		if h == name {
			return i
		}
	}
	fail("%s: no column named %q", t.path, name)
	return -1
}

func (t *table) get(row []string, name string) string { return row[t.col(name)] }

var problems []string

func fail(format string, a ...any) {
	problems = append(problems, fmt.Sprintf(format, a...))
}

func report() {
	fmt.Println("\nstructural problems:")
	for _, p := range problems {
		fmt.Println("  -", p)
	}
	os.Exit(1)
}

func readTable(path string) *table {
	f, err := os.Open(path)
	if err != nil {
		fail("%s: %v", path, err)
		return nil
	}
	defer f.Close()
	r := csv.NewReader(f)
	r.FieldsPerRecord = 0 // a ragged file is an error, which is the point
	rows, err := r.ReadAll()
	if err != nil {
		fail("%s: %v", path, err)
		return nil
	}
	if len(rows) < 2 {
		fail("%s: only %d rows", path, len(rows))
		return nil
	}
	seen := map[string]bool{}
	for _, h := range rows[0] {
		if seen[h] {
			fail("%s: duplicate column %q", path, h)
		}
		seen[h] = true
	}
	return &table{path: path, header: rows[0], rows: rows[1:]}
}

// Every field that parses as a number has to be finite, and every field in a
// column that is numeric everywhere else has to parse at all.
func checkNumeric(t *table) {
	for c, name := range t.header {
		numeric, total := 0, 0
		for _, row := range t.rows {
			v := strings.TrimSpace(row[c])
			if v == "" {
				fail("%s: empty value in column %q", t.path, name)
				continue
			}
			total++
			f, err := strconv.ParseFloat(v, 64)
			if err != nil {
				continue
			}
			numeric++
			if math.IsNaN(f) || math.IsInf(f, 0) {
				fail("%s: %s is %s", t.path, name, v)
			}
		}
		if numeric > 0 && numeric != total {
			fail("%s: column %q is numeric in %d of %d rows", t.path, name, numeric, total)
		}
	}
}

func set(t *table, name string) []string {
	seen := map[string]bool{}
	var out []string
	for _, row := range t.rows {
		v := t.get(row, name)
		if !seen[v] {
			seen[v] = true
			out = append(out, v)
		}
	}
	sort.Strings(out)
	return out
}

func sameSet(got, want []string) bool {
	if len(got) != len(want) {
		return false
	}
	w := append([]string(nil), want...)
	sort.Strings(w)
	for i := range got {
		if got[i] != w[i] {
			return false
		}
	}
	return true
}

func median(v []float64) float64 {
	s := append([]float64(nil), v...)
	sort.Float64s(s)
	n := len(s)
	if n%2 == 1 {
		return s[n/2]
	}
	return (s[n/2-1] + s[n/2]) / 2
}

func mustFloat(t *table, row []string, name string) float64 {
	f, err := strconv.ParseFloat(strings.TrimSpace(t.get(row, name)), 64)
	if err != nil {
		fail("%s: %q in column %s is not a number", t.path, t.get(row, name), name)
	}
	return f
}

func main() {
	root := flag.String("root", "../..", "repository root")
	outPath := flag.String("out", "", "where to write the recomputed medians")
	flag.Parse()
	res := filepath.Join(*root, "results")

	files, err := filepath.Glob(filepath.Join(res, "*.csv"))
	if err != nil || len(files) == 0 {
		fmt.Fprintln(os.Stderr, "gocheck: no CSVs under results/")
		os.Exit(1)
	}
	sort.Strings(files)
	tables := map[string]*table{}
	for _, p := range files {
		t := readTable(p)
		if t == nil {
			continue
		}
		checkNumeric(t)
		tables[filepath.Base(p)] = t
		fmt.Printf("  %-22s %d rows, %d columns, no ragged rows, no NaN or Inf\n",
			filepath.Base(p), len(t.rows), len(t.header))
	}

	nfe, straight := tables["nfe-quality.csv"], tables["straightness.csv"]
	if nfe == nil || straight == nil {
		fail("nfe-quality.csv or straightness.csv could not be read")
		report()
	}

	// "median of 3 seeds" in the README is only true if all three are there.
	for _, t := range []*table{nfe, straight, tables["training-curves.csv"]} {
		if t == nil {
			continue
		}
		if got := set(t, "seed"); !sameSet(got, wantSeeds) {
			fail("%s: seeds are %v, README says median of 3 seeds (%v)", t.path, got, wantSeeds)
		}
		if got := set(t, "model"); !sameSet(got, wantModels) {
			fail("%s: models are %v, expected %v", t.path, got, wantModels)
		}
		if got := set(t, "dataset"); !sameSet(got, wantData) {
			fail("%s: datasets are %v, expected %v", t.path, got, wantData)
		}
	}

	// Every published cell needs a full set of seeds behind it.
	counts := map[string]int{}
	for _, row := range nfe.rows {
		if nfe.get(row, "sampler") != "euler" {
			continue
		}
		counts[nfe.get(row, "dataset")+"/"+nfe.get(row, "model")+"/"+nfe.get(row, "nfe")]++
	}
	for _, ds := range wantData {
		for _, m := range wantModels {
			for _, n := range publishedNFE {
				k := fmt.Sprintf("%s/%s/%d", ds, m, n)
				if counts[k] != len(wantSeeds) {
					fail("nfe-quality.csv: %s euler has %d seeds, expected %d",
						k, counts[k], len(wantSeeds))
				}
			}
		}
	}

	// run-meta.json has to describe the run the CSVs actually contain.
	var meta struct {
		Datasets []string `json:"datasets"`
		Seeds    []int    `json:"seeds"`
	}
	if b, err := os.ReadFile(filepath.Join(res, "run-meta.json")); err != nil {
		fail("run-meta.json: %v", err)
	} else if err := json.Unmarshal(b, &meta); err != nil {
		fail("run-meta.json: %v", err)
	} else {
		var seeds []string
		for _, s := range meta.Seeds {
			seeds = append(seeds, strconv.Itoa(s))
		}
		if !sameSet(set(nfe, "dataset"), meta.Datasets) {
			fail("run-meta.json datasets %v do not match nfe-quality.csv %v",
				meta.Datasets, set(nfe, "dataset"))
		}
		if !sameSet(set(nfe, "seed"), seeds) {
			fail("run-meta.json seeds %v do not match nfe-quality.csv %v",
				meta.Seeds, set(nfe, "seed"))
		}
	}

	// The medians themselves, in the shared format verify.sh diffs.
	out := io.Discard
	if *outPath != "" {
		f, err := os.Create(*outPath)
		if err != nil {
			fmt.Fprintln(os.Stderr, "gocheck:", err)
			os.Exit(1)
		}
		defer f.Close()
		out = f
	}

	for _, ds := range wantData {
		for _, m := range wantModels {
			for _, n := range publishedNFE {
				var v []float64
				for _, row := range nfe.rows {
					if nfe.get(row, "dataset") == ds && nfe.get(row, "model") == m &&
						nfe.get(row, "sampler") == "euler" &&
						nfe.get(row, "nfe") == strconv.Itoa(n) {
						v = append(v, mustFloat(nfe, row, "sliced_w2"))
					}
				}
				if len(v) > 0 {
					fmt.Fprintf(out, "w2,%s,%s,%03d,%.10f\n", ds, m, n, median(v))
				}
			}
			for _, col := range []string{"straightness_S", "path_length_ratio_mean"} {
				var v []float64
				for _, row := range straight.rows {
					if straight.get(row, "dataset") == ds && straight.get(row, "model") == m {
						v = append(v, mustFloat(straight, row, col))
					}
				}
				if len(v) > 0 {
					fmt.Fprintf(out, "%s,%s,%s,median,%.10f\n", col, ds, m, median(v))
				}
			}
		}
	}

	if len(problems) > 0 {
		report()
	}
	fmt.Println("  all results files well formed, 3 seeds behind every published cell")
	os.Exit(0)
}
