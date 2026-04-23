/**
 * ============================================================
 * HARD-MARGIN SUPPORT VECTOR MACHINE — OPTİMAL SINIR ANALİZİ
 * ============================================================
 *
 * MAKSIMUM MARGIN PRENSİBİNİN GEOMETRİK ANLAMI:
 * -----------------------------------------------
 * İki sınıfı ayıran sonsuz sayıda hiper-düzlem (2D'de doğru) mevcuttur.
 * SVM, bu düzlemler arasından her iki sınıfa olan uzaklığı MAKSIMUM yapanı seçer.
 * Geometrik olarak: karar sınırı, iki sınıfın en yakın noktaları arasındaki
 * bölgenin tam ortasına konumlanır. Bu "en geniş sokak" (maximum margin hyperplane)
 * yaklaşımıdır.
 *
 * SUPPORT VECTOR'LERİN KRİTİK ROLÜ:
 * ------------------------------------
 * Karar sınırını yalnızca **support vector'ler** (margin üzerindeki noktalar)
 * belirler. Diğer tüm noktaları veri setinden kaldırsanız bile, karar sınırı
 * değişmez. Bu, modelin geri kalan noktaların gürültüsünden bağımsız olduğu
 * anlamına gelir. Dual formülasyonda yalnızca support vector'lerin Lagrange
 * çarpanı (alpha) > 0 olduğundan w = Σ αᵢ yᵢ xᵢ ifadesinde sadece onlar rol oynar.
 *
 * CONVEX OPTİMİZASYON VE GLOBAL OPTİMUM:
 * ----------------------------------------
 * Hard-margin SVM primal problemi:
 *   minimize   ½ ||w||²
 *   subject to yᵢ(w·xᵢ + b) ≥ 1,  ∀i
 *
 * Bu, konveks kısıtlı bir QP (Quadratic Programming) problemidir. Amaç fonksiyonu
 * konveks, kısıtlar ise afin (dolayısıyla konveks). Konveks bir problemde yerel
 * optimum = global optimum garantisi vardır. KKT koşulları hem gerekli hem yeterlidir.
 * SMO, bu dual QP'yi analitik alt-problem çözümleriyle iteratif olarak çözer.
 *
 * "GÜVENLİK KORİDORU" ANALOJİSİ:
 * ---------------------------------
 * Otonom araç navigasyonunda margin = araç için güvenli geçiş genişliği.
 * Karar sınırı = yolun merkez çizgisi.
 * Support vector'ler = yolun kenarlarındaki en kritik engel/sınır noktaları.
 * Margin ne kadar geniş → araç o kadar güvenli geçer.
 * Hard-margin SVM bu koridoru MAKSIMUM yapar; soft-margin ise bazı ihlallere
 * (engel çok yakınsa) tolerans gösterir.
 *
 * SMO ALGORITHM REFERENCE:
 *   Platt, J. (1998). Sequential Minimal Optimization: A Fast Algorithm for
 *   Training Support Vector Machines. Microsoft Research Technical Report.
 * ============================================================
 */

// ─── INTERFACES ─────────────────────────────────────────────────────────────

interface IDataPoint {
    readonly x: number;
    readonly y: number;
    readonly label: 1 | -1;
}

interface IDataset {
    readonly points: ReadonlyArray<IDataPoint>;
    readonly size: number;
    addPoint(point: IDataPoint): IDataset;
}

interface ISVMModel {
    readonly weights: Readonly<{ w1: number; w2: number }>;
    readonly bias: number;
    readonly margin: number;
    readonly supportVectors: ReadonlyArray<IDataPoint>;
    readonly alphas: Float64Array;
    decisionFunction(point: IDataPoint): number;
}

interface ISMOSolver {
    solve(dataset: IDataset): ISVMModel;
}

interface ISVMClassifier {
    train(dataset: IDataset): void;
    predict(point: IDataPoint): 1 | -1;
    getModel(): ISVMModel;
}

interface IConsoleVisualizer {
    printModel(model: ISVMModel): void;
    printPredictions(classifier: ISVMClassifier, testPoints: ReadonlyArray<IDataPoint>): void;
    printDatasetSummary(dataset: IDataset): void;
}

interface SMOConfig {
    readonly maxIterations: number;
    readonly tolerance: number;        // KKT violation tolerance
    readonly epsilon: number;          // alpha change tolerance
}

// ─── DATA POINT ─────────────────────────────────────────────────────────────

/**
 * Immutable value object representing a labeled 2D point.
 * Complexity: O(1) construction, O(1) access.
 */
class DataPoint implements IDataPoint {
    readonly x: number;
    readonly y: number;
    readonly label: 1 | -1;

    constructor(x: number, y: number, label: 1 | -1) {
        this.x = x;
        this.y = y;
        this.label = label;
    }

    /**
     * Dot product with another point treated as a vector.
     * O(1) — fixed 2D dimensionality.
     */
    dot(other: IDataPoint): number {
        return this.x * other.x + this.y * other.y;
    }

    toString(): string {
        return `(${this.x.toFixed(4)}, ${this.y.toFixed(4)}) [label=${this.label > 0 ? '+1' : '-1'}]`;
    }
}

// ─── DATASET ────────────────────────────────────────────────────────────────

/**
 * Immutable-style collection of DataPoints.
 * addPoint returns a NEW Dataset (persistent/functional pattern).
 * Complexity: O(n) storage, O(1) amortized append via spread.
 */
class Dataset implements IDataset {
    readonly points: ReadonlyArray<IDataPoint>;

    constructor(points: ReadonlyArray<IDataPoint> = []) {
        this.points = points;
    }

    get size(): number {
        return this.points.length;
    }

    /** O(n) — creates new array with appended element */
    addPoint(point: IDataPoint): Dataset {
        return new Dataset([...this.points, point]);
    }

    /** Returns points of a specific class. O(n) */
    filterByLabel(label: 1 | -1): ReadonlyArray<IDataPoint> {
        return this.points.filter(p => p.label === label);
    }
}

// ─── DETERMINISTIC PSEUDO-RANDOM NUMBER GENERATOR ───────────────────────────

/**
 * Mulberry32 PRNG — fast, deterministic, seedable.
 * Produces uniform [0, 1) floats.
 * O(1) per call.
 */
function createSeededRng(seed: number): () => number {
    let s = seed;
    return (): number => {
        s |= 0;
        s = (s + 0x6d2b79f5) | 0;
        let t = Math.imul(s ^ (s >>> 15), 1 | s);
        t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
        return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
}

// ─── DATA GENERATION ────────────────────────────────────────────────────────

/**
 * Generates a linearly separable 2D dataset:
 *   Class +1: centered at (2, 2)  with Gaussian noise σ=0.5
 *   Class -1: centered at (-2,-2) with Gaussian noise σ=0.5
 *
 * Uses Box-Muller transform for Gaussian sampling.
 *
 * Complexity: O(n) — linear in number of points per class.
 * Memory:     O(n) DataPoint objects on the heap.
 */
function generateLinearlySeperableData(
    pointsPerClass: number = 20,
    seed: number = 42
): Dataset {
    const rng = createSeededRng(seed);
    const sigma = 0.5;
    const points: IDataPoint[] = [];

    // Box-Muller transform: two uniform samples → two standard normal samples
    // O(1) per pair
    const gaussianPair = (mu: number): [number, number] => {
        const u1 = rng();
        const u2 = rng();
        const mag = sigma * Math.sqrt(-2.0 * Math.log(u1 + 1e-10));
        return [
            mu + mag * Math.cos(2 * Math.PI * u2),
            mu + mag * Math.sin(2 * Math.PI * u2),
        ];
    };

    // O(n) — generate positive class around (+2, +2)
    for (let i = 0; i < pointsPerClass; i++) {
        const [x] = gaussianPair(2);
        const [y] = gaussianPair(2);
        points.push(new DataPoint(x, y, 1));
    }

    // O(n) — generate negative class around (-2, -2)
    for (let i = 0; i < pointsPerClass; i++) {
        const [x] = gaussianPair(-2);
        const [y] = gaussianPair(-2);
        points.push(new DataPoint(x, y, -1));
    }

    return new Dataset(points);
}

// ─── SVM MODEL ──────────────────────────────────────────────────────────────

/**
 * Immutable model snapshot produced by the solver.
 * Once created, the model is fully frozen — no mutation possible.
 */
class SVMModel implements ISVMModel {
    readonly weights: Readonly<{ w1: number; w2: number }>;
    readonly bias: number;
    readonly margin: number;
    readonly supportVectors: ReadonlyArray<IDataPoint>;
    readonly alphas: Float64Array;

    constructor(
        w1: number,
        w2: number,
        bias: number,
        supportVectors: ReadonlyArray<IDataPoint>,
        alphas: Float64Array
    ) {
        this.weights = Object.freeze({ w1, w2 });
        this.bias = bias;
        // margin = 2 / ||w||
        // O(1) — fixed 2D norm computation
        const wNorm = Math.sqrt(w1 * w1 + w2 * w2);
        this.margin = wNorm > 1e-10 ? 2.0 / wNorm : 0;
        this.supportVectors = Object.freeze([...supportVectors]);
        this.alphas = alphas;
        Object.freeze(this);
    }

    /**
     * Raw decision function value: f(x) = w·x + b
     * O(1) in 2D.
     */
    decisionFunction(point: IDataPoint): number {
        return this.weights.w1 * point.x + this.weights.w2 * point.y + this.bias;
    }
}

// ─── SMO SOLVER ─────────────────────────────────────────────────────────────

/**
 * Sequential Minimal Optimization (Platt, 1998)
 * ------------------------------------------------
 * Decomposes the QP into the smallest possible sub-problems:
 * at each step, exactly 2 alpha variables are optimized analytically.
 *
 * Dual Problem:
 *   maximize   Σᵢ αᵢ  −  ½ Σᵢ Σⱼ αᵢ αⱼ yᵢ yⱼ K(xᵢ,xⱼ)
 *   subject to Σᵢ αᵢ yᵢ = 0,   0 ≤ αᵢ  (hard-margin: no upper bound C)
 *
 * KKT conditions (hard-margin):
 *   αᵢ = 0   ⟹  yᵢ f(xᵢ) ≥ 1
 *   αᵢ > 0   ⟹  yᵢ f(xᵢ) = 1  (support vector)
 *
 * Mutable state is scoped entirely within this class during solve().
 * The returned SVMModel is immutable.
 */
class SMOSolver implements ISMOSolver {
    private readonly config: SMOConfig;

    constructor(config: SMOConfig) {
        this.config = config;
    }

    /**
     * Linear kernel: K(xᵢ, xⱼ) = xᵢ · xⱼ
     * O(1) in 2D.
     */
    private kernelLinear(a: IDataPoint, b: IDataPoint): number {
        return a.x * b.x + a.y * b.y;
    }

    /**
     * Precompute full kernel matrix K[i][j] = K(xᵢ, xⱼ).
     *
     * Complexity: O(n²) time, O(n²) space.
     * Memory:     Float64Array vs number[]
     *   - Float64Array: 8 bytes/element, contiguous C-style buffer → cache-friendly
     *   - number[]:    ~48–72 bytes/element (V8 boxed double + pointer overhead)
     *   For n=40: Float64Array ≈ 12.8 KB vs number[] ≈ ~115 KB
     *   After precomputation, every kernel lookup is O(1) (flat index arithmetic).
     */
    private buildKernelMatrix(points: ReadonlyArray<IDataPoint>): Float64Array {
        const n = points.length;
        // Flattened row-major matrix: index(i,j) = i*n + j
        const K = new Float64Array(n * n);
        for (let i = 0; i < n; i++) {
            for (let j = i; j < n; j++) {
                const val = this.kernelLinear(points[i], points[j]);
                K[i * n + j] = val;
                K[j * n + i] = val; // symmetric
            }
        }
        return K;
    }

    /**
     * Compute the SVM output for sample i using current alphas and bias.
     * f(xᵢ) = Σⱼ αⱼ yⱼ K(xⱼ,xᵢ) + b
     *
     * Complexity: O(n) per call — iterates all training points.
     * Called inside the SMO loop → total across all iterations: O(n² * maxIter).
     */
    private computeOutput(
        i: number,
        alphas: Float64Array,
        labels: Float64Array,
        K: Float64Array,
        b: number,
        n: number
    ): number {
        let sum = 0;
        for (let j = 0; j < n; j++) {
            if (alphas[j] > 0) {
                sum += alphas[j] * labels[j] * K[j * n + i];
            }
        }
        return sum + b;
    }

    /**
     * SMO Main Loop
     * ──────────────
     * Worst case:  O(n² × maxIterations)
     *   Each full pass over n points calls computeOutput → O(n) per point.
     *   Up to maxIterations passes ⟹ O(n² × T).
     *
     * Average case: O(n × T) — many alpha pairs don't violate KKT ⟹ skipped early.
     *
     * The alpha pair (i, j) selection heuristic:
     *   - Outer loop: scan all points for KKT violations (passes entire dataset).
     *   - Inner selection: pick j ≠ i maximizing |Eᵢ - Eⱼ| (maximum step heuristic).
     */
    solve(dataset: IDataset): ISVMModel {
        const points = dataset.points;
        const n = points.length;
        const { maxIterations, tolerance, epsilon } = this.config;

        // ── Mutable solver state (scoped here, not leaked outside) ──────────────
        const alphas = new Float64Array(n);         // Lagrange multipliers, O(n) space
        const labels = new Float64Array(n);         // y_i ∈ {+1, -1}
        const errors = new Float64Array(n);         // error cache: Eᵢ = f(xᵢ) - yᵢ
        let b = 0;                                  // bias term

        for (let i = 0; i < n; i++) {
            labels[i] = points[i].label;
        }

        // O(n²) kernel matrix precomputation
        const K = this.buildKernelMatrix(points);

        let iter = 0;
        let numChangedAlphas = 0;
        let examineAll = true;

        // ── SMO Main Loop ────────────────────────────────────────────────────────
        // Loop continues while alphas are changing or we haven't done a full pass.
        // Worst case: O(n² × maxIterations)
        while (iter < maxIterations && (numChangedAlphas > 0 || examineAll)) {
            numChangedAlphas = 0;

            // Outer loop: iterate over working set
            const range = examineAll
                ? Array.from({ length: n }, (_, k) => k)
                : Array.from({ length: n }, (_, k) => k).filter(k => alphas[k] > 0 && alphas[k] < 1e15);

            for (const i of range) {
                // Compute output and error for i — O(n)
                const fi = this.computeOutput(i, alphas, labels, K, b, n);
                const Ei = fi - labels[i];
                errors[i] = Ei;

                const ri = Ei * labels[i]; // KKT residual

                // Check KKT violation: ri < -tol (α should be larger)
                //                    or ri > tol  (α should be smaller)
                const kktViolated =
                    (ri < -tolerance && alphas[i] < 1e15) ||
                    (ri > tolerance && alphas[i] > 0);

                if (!kktViolated) continue;

                // ── Inner loop: select j ≠ i by maximum step heuristic ───────────
                let j = -1;
                let maxDeltaE = 0;

                // First try cached errors to maximize |Ei - Ej|
                for (let k = 0; k < n; k++) {
                    if (k === i) continue;
                    const delta = Math.abs(Ei - errors[k]);
                    if (delta > maxDeltaE) {
                        maxDeltaE = delta;
                        j = k;
                    }
                }

                if (j === -1) {
                    // Fallback: random selection
                    j = (i + 1 + Math.floor(Math.random() * (n - 1))) % n;
                }

                // ── Optimize alpha pair (i, j) analytically ─────────────────────
                const fj = this.computeOutput(j, alphas, labels, K, b, n);
                const Ej = fj - labels[j];
                errors[j] = Ej;

                const alphaIOld = alphas[i];
                const alphaJOld = alphas[j];
                const yi = labels[i];
                const yj = labels[j];

                // Compute bounds L and H for alpha_j (hard-margin: C = ∞, so only L matters)
                // From constraint: Σ αᵢ yᵢ = 0 and αᵢ ≥ 0
                let L: number, H: number;
                if (yi === yj) {
                    // Same class: α_i + α_j = constant ≥ 0 → L ≥ 0
                    L = Math.max(0, alphaIOld + alphaJOld - 1e15);
                    H = alphaIOld + alphaJOld;
                } else {
                    // Different classes: α_j - α_i = constant
                    L = Math.max(0, alphaJOld - alphaIOld);
                    H = 1e15; // unbounded in hard-margin
                }

                if (L >= H - epsilon) continue;

                // Second derivative of objective along α_j
                // η = K(xi,xi) + K(xj,xj) - 2*K(xi,xj)
                // Must be > 0 for a valid step (positive definite kernel)
                const eta = K[i * n + i] + K[j * n + j] - 2.0 * K[i * n + j];
                if (eta <= 0) continue;

                // Unconstrained update for alpha_j
                let alphaJNew = alphaJOld + (yj * (Ei - Ej)) / eta;

                // Clip to [L, H]
                alphaJNew = Math.min(Math.max(alphaJNew, L), H);

                if (Math.abs(alphaJNew - alphaJOld) < epsilon) continue;

                // Update alpha_i via equality constraint: Σ αᵢ yᵢ = 0
                const alphaINew = alphaIOld + yi * yj * (alphaJOld - alphaJNew);
                if (alphaINew < 0) continue; // hard-margin guard

                alphas[i] = alphaINew;
                alphas[j] = alphaJNew;

                // ── Update bias b ─────────────────────────────────────────────────
                // From KKT: if 0 < αᵢ → yᵢ f(xᵢ) = 1 → b is determined
                const Kii = K[i * n + i];
                const Kjj = K[j * n + j];
                const Kij = K[i * n + j];

                const b1 =
                    b -
                    Ei -
                    yi * (alphaINew - alphaIOld) * Kii -
                    yj * (alphaJNew - alphaJOld) * Kij;

                const b2 =
                    b -
                    Ej -
                    yi * (alphaINew - alphaIOld) * Kij -
                    yj * (alphaJNew - alphaJOld) * Kjj;

                if (alphaINew > epsilon && alphaINew < 1e14) {
                    b = b1;
                } else if (alphaJNew > epsilon && alphaJNew < 1e14) {
                    b = b2;
                } else {
                    b = (b1 + b2) / 2.0;
                }

                // Invalidate error cache for i and j
                errors[i] = this.computeOutput(i, alphas, labels, K, b, n) - labels[i];
                errors[j] = this.computeOutput(j, alphas, labels, K, b, n) - labels[j];

                numChangedAlphas++;
            }

            iter++;
            // Alternate between full pass and active-set pass
            if (examineAll) {
                examineAll = false;
            } else if (numChangedAlphas === 0) {
                examineAll = true;
            }
        }

        // ── Reconstruct w from dual solution ─────────────────────────────────
        // w = Σᵢ αᵢ yᵢ xᵢ  (support vector expansion)
        // O(n) — iterate all alphas
        let w1 = 0, w2 = 0;
        for (let i = 0; i < n; i++) {
            if (alphas[i] > epsilon) {
                w1 += alphas[i] * labels[i] * points[i].x;
                w2 += alphas[i] * labels[i] * points[i].y;
            }
        }

        // ── Identify support vectors ──────────────────────────────────────────
        // O(n) — filter points where alpha > 0
        const supportVectors: IDataPoint[] = [];
        for (let i = 0; i < n; i++) {
            if (alphas[i] > epsilon) {
                supportVectors.push(points[i]);
            }
        }

        // Recompute b from support vectors for numerical stability
        // O(sv) where sv = number of support vectors
        let bSum = 0;
        let svCount = 0;
        for (let i = 0; i < n; i++) {
            if (alphas[i] > epsilon) {
                let fi = 0;
                for (let j = 0; j < n; j++) {
                    if (alphas[j] > epsilon) {
                        fi += alphas[j] * labels[j] * K[j * n + i];
                    }
                }
                bSum += labels[i] - fi;
                svCount++;
            }
        }
        const bFinal = svCount > 0 ? bSum / svCount : b;

        // Return frozen model
        return new SVMModel(w1, w2, bFinal, supportVectors, alphas);
    }
}

// ─── SVM CLASSIFIER (FACADE) ─────────────────────────────────────────────────

/**
 * Facade combining Dataset + SMOSolver into a clean train/predict API.
 *
 * Dependency injection via constructor ensures testability and decoupling.
 */
class SVMClassifier implements ISVMClassifier {
    private readonly solver: ISMOSolver;
    private model: ISVMModel | null = null;

    constructor(solver: ISMOSolver) {
        this.solver = solver;
    }

    /** O(n² × maxIterations) — delegates entirely to SMOSolver */
    train(dataset: IDataset): void {
        if (dataset.size < 2) {
            throw new Error('Dataset must contain at least 2 points.');
        }
        this.model = this.solver.solve(dataset);
    }

    /**
     * Classify a single point.
     *
     * f(x) = Σ_{sv} αᵢ yᵢ K(xᵢ, x) + b
     *
     * Complexity: O(|SV|) — iterate only support vectors (alphas[i] > 0).
     * This is the key efficiency advantage over O(n) naive re-evaluation:
     * only support vectors contribute to the decision function.
     */
    predict(point: IDataPoint): 1 | -1 {
        if (!this.model) throw new Error('Model not trained. Call train() first.');
        const val = this.model.decisionFunction(point);
        return val >= 0 ? 1 : -1;
    }

    getModel(): ISVMModel {
        if (!this.model) throw new Error('Model not trained. Call train() first.');
        return this.model;
    }
}

// ─── CONSOLE VISUALIZER ──────────────────────────────────────────────────────

/**
 * Structured terminal output for model inspection and prediction reporting.
 * Pure output concern — no computation.
 * All methods are O(n) or O(|SV|) in their respective data sizes.
 */
class ConsoleVisualizer implements IConsoleVisualizer {
    private readonly width: number;

    constructor(width: number = 70) {
        this.width = width;
    }

    private separator(char: string = '─'): void {
        console.log(char.repeat(this.width));
    }

    private header(title: string): void {
        this.separator('═');
        const pad = Math.floor((this.width - title.length) / 2);
        console.log(' '.repeat(pad) + title);
        this.separator('═');
    }

    private section(title: string): void {
        this.separator('─');
        console.log(`  ▶  ${title}`);
        this.separator('─');
    }

    printDatasetSummary(dataset: IDataset): void {
        this.header('📊  VERİ SETİ ÖZETİ');
        const pos = dataset.points.filter(p => p.label === 1).length;
        const neg = dataset.points.filter(p => p.label === -1).length;
        console.log(`  Toplam nokta  : ${dataset.size}`);
        console.log(`  Sınıf +1      : ${pos} nokta`);
        console.log(`  Sınıf -1      : ${neg} nokta`);
        console.log();
    }

    printModel(model: ISVMModel): void {
        this.header('🤖  EĞİTİLMİŞ SVM MODELİ');

        const { w1, w2 } = model.weights;
        const b = model.bias;

        // ── Karar Sınırı ────────────────────────────────────────────────────────
        this.section('Karar Sınırı: w·x + b = 0');
        const w1s = w1 >= 0 ? `+${w1.toFixed(6)}` : w1.toFixed(6);
        const w2s = w2 >= 0 ? `+${w2.toFixed(6)}` : w2.toFixed(6);
        const bs = b >= 0 ? `+${b.toFixed(6)}` : b.toFixed(6);
        console.log(`  Denklem : (${w1s})·x  (${w2s})·y  (${bs}) = 0`);
        console.log();

        // Normalleştirilmiş ax + by + c = 0 formu
        const norm = Math.sqrt(w1 * w1 + w2 * w2);
        console.log(`  Normalize (birim normal): (${(w1 / norm).toFixed(4)})x + (${(w2 / norm).toFixed(4)})y + (${(b / norm).toFixed(4)}) = 0`);
        console.log();

        // ── Ağırlık Vektörü ve Margin ─────────────────────────────────────────────
        this.section('Ağırlık Vektörü ve Margin');
        console.log(`  w  = [${w1.toFixed(6)}, ${w2.toFixed(6)}]`);
        console.log(`  b  = ${b.toFixed(6)}`);
        console.log(`  ||w|| = ${norm.toFixed(6)}`);
        console.log(`  Margin = 2 / ||w|| = ${model.margin.toFixed(6)}`);
        console.log();

        // ── Destek Vektörleri ────────────────────────────────────────────────────
        this.section(`Destek Vektörleri (${model.supportVectors.length} adet bulundu)`);
        if (model.supportVectors.length === 0) {
            console.log('  ⚠  Destek vektörü bulunamadı. Yakınsamayı kontrol edin.');
        } else {
            model.supportVectors.forEach((sv, idx) => {
                const labelStr = sv.label === 1 ? '[ +1 ]' : '[ -1 ]';
                const dist = Math.abs(w1 * sv.x + w2 * sv.y + b) / norm;
                console.log(
                    `  SV[${String(idx).padStart(2, '0')}] ${labelStr}  x=${sv.x.toFixed(4).padStart(9)}  y=${sv.y.toFixed(4).padStart(9)}  margin-dist=${dist.toFixed(4)}`
                );
            });
        }
        console.log();

        // ── ASCII Visualization ────────────────────────────────────────────────
        this.printASCIIPlot(model);
    }

    private printASCIIPlot(model: ISVMModel): void {
        this.section('ASCII Karar Sınırı Grafiği  [-4, 4] × [-4, 4]');
        const W = 50, H = 22;
        const xMin = -4, xMax = 4, yMin = -4, yMax = 4;
        const { w1, w2 } = model.weights;
        const b = model.bias;

        // Build char grid
        const grid: string[][] = Array.from({ length: H }, () => Array(W).fill(' '));

        for (let row = 0; row < H; row++) {
            for (let col = 0; col < W; col++) {
                const x = xMin + (col / (W - 1)) * (xMax - xMin);
                const y = yMax - (row / (H - 1)) * (yMax - yMin);
                const val = w1 * x + w2 * y + b;
                grid[row][col] = val > 0 ? '·' : '░';
            }
        }

        // Draw decision boundary (zero crossing)
        for (let col = 0; col < W; col++) {
            const x = xMin + (col / (W - 1)) * (xMax - xMin);
            for (let row = 0; row < H - 1; row++) {
                const y0 = yMax - (row / (H - 1)) * (yMax - yMin);
                const y1 = yMax - ((row + 1) / (H - 1)) * (yMax - yMin);
                if ((w1 * x + w2 * y0 + b) * (w1 * x + w2 * y1 + b) <= 0) {
                    grid[row][col] = '│';
                }
            }
        }

        // Mark support vectors
        model.supportVectors.forEach(sv => {
            const col = Math.round(((sv.x - xMin) / (xMax - xMin)) * (W - 1));
            const row = Math.round(((yMax - sv.y) / (yMax - yMin)) * (H - 1));
            if (row >= 0 && row < H && col >= 0 && col < W) {
                grid[row][col] = sv.label === 1 ? '◆' : '◇';
            }
        });

        // Print with y-axis labels
        for (let row = 0; row < H; row++) {
            const y = yMax - (row / (H - 1)) * (yMax - yMin);
            const yLabel = row % 5 === 0 ? y.toFixed(1).padStart(5) : '     ';
            console.log(`  ${yLabel} │ ${grid[row].join('')}`);
        }
        console.log(`         └${'─'.repeat(W + 1)}`);
        console.log(`           ${xMin.toFixed(0)}${' '.repeat(Math.floor(W / 2) - 3)}0${' '.repeat(Math.floor(W / 2) - 2)}${xMax.toFixed(0)}`);
        console.log();
        console.log('  Lejant: ·=Sınıf +1 bölgesi  ░=Sınıf -1 bölgesi  │=Karar sınırı');
        console.log('          ◆=Destek vektörü(+1)  ◇=Destek vektörü(-1)');
        console.log();
    }

    printPredictions(
        classifier: ISVMClassifier,
        testPoints: ReadonlyArray<IDataPoint>
    ): void {
        this.header('🔍  TAHMİN SONUÇLARI');
        console.log(`  ${'Nokta'.padEnd(28)} ${'Gerçek'.padEnd(6)} ${'Tahmin'.padEnd(6)} ${'Karar f(x)'.padEnd(14)} Sonuç`);
        this.separator('─');

        let correct = 0;
        const model = classifier.getModel();

        // O(|testPoints| × |SV|) — predict() is O(|SV|) per point
        testPoints.forEach((pt, idx) => {
            const pred = classifier.predict(pt);
            const fval = model.decisionFunction(pt);
            const match = pred === pt.label;
            if (match) correct++;

            const trueStr = pt.label === 1 ? '+1' : '-1';
            const predStr = pred === 1 ? '+1' : '-1';
            const fStr = fval.toFixed(4);
            const ptStr = `(${pt.x.toFixed(3)}, ${pt.y.toFixed(3)})`.padEnd(28);
            const status = match ? '✓' : '✗';
            console.log(`  ${ptStr} ${trueStr.padEnd(6)} ${predStr.padEnd(6)} ${fStr.padEnd(14)} ${status}`);

            if ((idx + 1) % 10 === 0 && idx < testPoints.length - 1) {
                this.separator('·');
            }
        });

        this.separator('─');
        const acc = ((correct / testPoints.length) * 100).toFixed(1);
        console.log(`  Doğruluk: ${correct} / ${testPoints.length}  (%${acc})`);
        console.log();
    }

    printComplexityAnalysis(): void {
        this.header('⏱  KARMAŞIKLIK ANALİZİ');
        const rows: [string, string, string][] = [
            ['generateLinearlySeperableData(n)', 'O(n)', 'n = toplam nokta sayısı'],
            ['buildKernelMatrix(n)', 'O(n²)', 'bir kez ön hesaplanır'],
            ['computeOutput(i)', 'O(n)', 'tüm αⱼ > 0 üzerinden'],
            ['SMO ana döngü (en kötü)', 'O(n² × T)', 'T = maxIterations'],
            ['SMO ana döngü (ortalama)', 'O(n × T)', 'seyrek aktif küme'],
            ['predict(x)', 'O(|SV|)', 'genellikle |SV| ≪ n'],
            ['Bellek — Float64Array n²', 'O(n²) × 8B', 'n=40 için: ~12.8 KB'],
            ['Bellek — number[] n²', 'O(n²) × ~56B', 'n=40 için: ~90 KB'],
        ];
        rows.forEach(([fn, complexity, note]) => {
            console.log(
                `  ${fn.padEnd(38)} ${complexity.padEnd(16)} // ${note}`
            );
        });
        console.log();
    }
}

// ─── MAIN DEMONSTRATION ──────────────────────────────────────────────────────

/**
 * main() orchestrates the full pipeline:
 *  1. Generate dataset
 *  2. Train SVMClassifier via SMOSolver
 *  3. Print model diagnostics
 *  4. Run predictions on held-out test points
 *  5. Print complexity analysis
 *
 * Total complexity: O(n² × maxIterations) dominated by SMO training.
 */
function main(): void {
    console.log('\n');

    // ── 1. Generate training data ──────────────────────────────────────────
    const trainDataset = generateLinearlySeperableData(20, 42);

    // ── 2. Configure and build classifier ─────────────────────────────────
    const smoConfig: SMOConfig = {
        maxIterations: 1000,
        tolerance: 1e-3,   // KKT violation tolerance
        epsilon: 1e-6,     // minimum meaningful alpha change
    };

    const solver = new SMOSolver(smoConfig);
    const classifier = new SVMClassifier(solver);
    const visualizer = new ConsoleVisualizer(72);

    visualizer.printDatasetSummary(trainDataset);

    // ── 3. Eğitim ─────────────────────────────────────────────────────────
    console.log('  ⚙  SVM Eğitiliyor (SMO çözücü)...');
    const t0 = Date.now();
    classifier.train(trainDataset);
    const elapsed = Date.now() - t0;
    console.log(`  ✔  Eğitim ${elapsed} ms içinde tamamlandı\n`);

    // ── 4. Modeli Yazdır ──────────────────────────────────────────────────
    visualizer.printModel(classifier.getModel());

    // ── 5. Test noktaları üzerinde tahmin (aynı dağılım, farklı seed) ────
    const testDataset = generateLinearlySeperableData(10, 137);
    visualizer.printPredictions(classifier, testDataset.points);

    // ── 6. Özel "navigasyon" senaryosu noktalarında tahmin ────────────────
    const navigationPoints: IDataPoint[] = [
        new DataPoint(3.0, 3.0, 1),   // clearly in +1 zone
        new DataPoint(-3.0, -3.0, -1),   // clearly in -1 zone
        new DataPoint(0.5, 0.5, 1),   // near boundary, +1 side
        new DataPoint(-0.5, -0.5, -1),   // near boundary, -1 side
        new DataPoint(1.2, 1.8, 1),   // moderate +1
        new DataPoint(-1.5, -1.0, -1),   // moderate -1
    ];

    const navVisualizer = new ConsoleVisualizer(72);
    navVisualizer.printPredictions(classifier, navigationPoints);

    // ── 7. Complexity analysis ────────────────────────────────────────────
    visualizer.printComplexityAnalysis();

    // ── 8. Summary banner ─────────────────────────────────────────────────
    const model = classifier.getModel();
    const { w1, w2 } = model.weights;
    const b = model.bias;

    console.log('═'.repeat(72));
    console.log('  📌  FİNAL ÖZET');
    console.log('═'.repeat(72));
    console.log(`  Karar sınırı      : (${w1.toFixed(4)})x + (${w2.toFixed(4)})y + (${b.toFixed(4)}) = 0`);
    console.log(`  Margin genişliği  : ${model.margin.toFixed(6)}`);
    console.log(`  Destek vektörleri : ${model.supportVectors.length} adet`);
    console.log(`  Eğitim süresi     : ${elapsed} ms`);
    console.log(`  Konfigürasyon     : maxIter=${smoConfig.maxIterations}, tol=${smoConfig.tolerance}`);
    console.log();
    console.log('  🚗  Otonom Navigasyon Analojisi:');
    console.log(`  Güvenlik koridoru genişliği = ${model.margin.toFixed(4)} birim`);
    console.log(`  Tüm koridoru ${model.supportVectors.length} kritik sınır noktası tanımlıyor`);
    console.log('  Maksimum margin ⟹  maksimum güvenlik — otonom araçlar için optimal çözüm');
    console.log('═'.repeat(72));
    console.log();
}

main();