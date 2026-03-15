@preconcurrency import MLX
import MLXFast

/// A dynamically generated fused Metal kernel.
///
/// Created by the compiler from IR operation sequences via `MetalCodeGenerator`.
/// Wraps `MLXFast.metalKernel()` with the generated source code and
/// provides a type-safe call interface.
///
/// ## Lifecycle
///
/// 1. Compiler identifies fusable operation sequence in IR
/// 2. `MetalCodeGenerator.generate()` emits Metal source
/// 3. `FusedKernel.init()` compiles via `MLXFast.metalKernel()`
/// 4. `call()` dispatches the kernel with runtime inputs
///
/// The kernel is compiled once at model load time and reused for all
/// subsequent inference steps.
public struct FusedKernel: @unchecked Sendable {

    /// The compiled MLX Metal kernel.
    private let kernel: MLXFast.MLXFastKernel

    /// Input buffer names (for documentation/debugging).
    public let inputNames: [String]

    /// Output buffer names.
    public let outputNames: [String]

    /// Template parameters applied at call time.
    public struct TemplateParams: Sendable {
        public let hiddenSize: Int?
        public let kernelSize: Int?

        public init(hiddenSize: Int? = nil, kernelSize: Int? = nil) {
            self.hiddenSize = hiddenSize
            self.kernelSize = kernelSize
        }
    }

    /// Template parameters for this kernel.
    public let templateParams: TemplateParams

    /// Initialize a fused kernel from generated Metal source.
    ///
    /// - Parameters:
    ///   - name: unique kernel name (for MLX kernel cache)
    ///   - ops: fusable operation sequence
    ///   - inputNames: names matching the input buffers
    ///   - outputNames: names matching the output buffers
    ///   - templateParams: compile-time constants
    public init(
        name: String,
        ops: [MetalCodeGenerator.FusableOp],
        inputNames: [String],
        outputNames: [String],
        templateParams: TemplateParams = TemplateParams()
    ) {
        self.inputNames = inputNames
        self.outputNames = outputNames
        self.templateParams = templateParams

        let source = MetalCodeGenerator.generate(
            ops: ops, inputNames: inputNames, outputNames: outputNames)

        self.kernel = MLXFast.metalKernel(
            name: name,
            inputNames: inputNames,
            outputNames: outputNames,
            source: source,
            ensureRowContiguous: true
        )
    }

    /// Execute the fused kernel.
    ///
    /// - Parameters:
    ///   - inputs: input MLXArrays (flattened to 1D)
    ///   - gridSize: number of threads (typically hiddenSize)
    ///   - outputShapes: shapes for each output
    ///   - dtype: compute dtype
    /// - Returns: output MLXArrays
    public func call(
        inputs: [MLXArray],
        gridSize: Int,
        outputShapes: [[Int]],
        dtype: DType
    ) -> [MLXArray] {
        var template: [(String, any KernelTemplateArg)] = [("T", dtype)]
        if let D = templateParams.hiddenSize {
            template.append(("HIDDEN_D", D))
        }
        if let K = templateParams.kernelSize {
            template.append(("CONV_K", K))
        }

        return kernel(
            inputs,
            template: template,
            grid: (gridSize, 1, 1),
            threadGroup: (min(gridSize, 1024), 1, 1),
            outputShapes: outputShapes,
            outputDTypes: outputShapes.map { _ in dtype }
        )
    }
}
